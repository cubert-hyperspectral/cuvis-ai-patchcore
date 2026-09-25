# cuvis-ai-patchcore

HSI-PatchCore for [cuvis-ai](https://docs.cuvis.ai/latest/): a nonparametric memory-bank anomaly
detector that scores every pixel of a hyperspectral cube by the distance between its (locally
averaged, per-band standardised) spectrum and the nearest spectrum of a coreset built from normal
data. It is PatchCore (Roth et al., CVPR 2022) with the CNN patch features replaced by the raw
spectrum — hyperspectral pixels are already dense, physically meaningful features, so no backbone
is needed, nothing is gradient-trained, and the whole model is a fitted `(coreset_size, C)` buffer.

Where a global Gaussian (RX) has to model a multimodal "normal" (several materials on a belt) with
one covariance, the memory bank keeps one entry per normal spectral mode, so in-distribution
mixtures stop firing while out-of-distribution spectra (foreign objects, unknown materials) stay far
from every entry.

The plugin ships four nodes:

| Node | Role |
|---|---|
| [`PatchCoreDetector`](#patchcoredetector) | memory-bank detector on spectra or any dense feature grid (Phase-1 fitted) |
| [`ScoreRangeNormalizer`](#scorerangenormalizer) | puts one detector's map on its normal range before fusion (Phase-1 fitted) |
| [`ScoreMapFusion`](#scoremapfusion) | fuses N maps: mean, min, max, weighted mean, or a priority rule for gated maps |
| [`FrameScoreGate`](#framescoregate) | blanks a map on frames whose top-k score stays at or below a threshold |

Requires `cuvis-ai-core >= 0.17.4` and `cuvis-ai-schemas >= 0.12.0` on Python 3.11 – 3.13.

## PatchCoreDetector

`cuvis_ai_patchcore.node.patchcore.PatchCoreDetector`

| Port | Direction | Shape / dtype | Notes |
|---|---|---|---|
| `cube` | in | `[B, H, W, C]` float32 | a hyperspectral cube or any dense feature grid (e.g. ViT patch tokens); `C == input_channels` |
| `reference` | in, optional | `[B, H_ref, W_ref, *]` float32 | its spatial size sets the `scores` resolution (pass the cube when the input is a coarse feature grid) |
| `scores` | out | `[B, H, W, 1]` float32 | nearest-coreset Euclidean distance, bilinearly upsampled from the stride grid to the input (or reference) size |
| `anomaly_score` | out | `[B]` float32 | mean of the top `max(1, floor(topk_frac · H · W))` pixel scores (image-level alarm) |

| hparam | default | meaning |
|---|---|---|
| `input_channels` | — | spectral bands `C` (sizes the buffers eagerly) |
| `coreset_size` | 8000 | memory-bank rows after k-center greedy (fixed buffer; a smaller bank is cycled) |
| `stride` | 4 | scoring grid stride; latency scales ~ 1/stride² |
| `bank_stride` | 10 | grid stride while collecting normal features in Phase 1 |
| `pool_size` | 3 | odd local averaging window (`1` = no pooling) |
| `max_bank_size` | 80000 | seeded random cap on collected features before coreset selection |
| `topk_frac` | 0.001 | pixel fraction averaged into `anomaly_score` |
| `chunk_size` | 4096 | query rows per `torch.cdist` call |
| `autocast_dtype` | `null` | `float16` / `bfloat16` nearest-neighbour search on CUDA (CPU stays float32) |
| `standardize` | `true` | z-score every channel with fitted statistics (spectra); `false` uses the input features unchanged (deep feature grids) |
| `seed` | 0 | RNG seed for the bank cap, the greedy start point and the projection |
| `coreset_projection` | `null` | `null` selects the coreset on exact distances; `sparse_random` runs anomalib's sparse random projection (Johnson-Lindenstrauss dimension for `projection_eps`) before k-center greedy, i.e. anomalib's `KCenterGreedy` recipe. Only the row selection changes; scoring always uses the original features |
| `projection_eps` | 0.9 | JL distortion parameter of the projection (anomalib default) |
| `eps` | 1e-6 | floor for the per-band standard deviation |

Fitted state (`mu`, `sd`, `coreset`) lives in buffers and is written to the pipeline `.pt` by
`save_to_file`; loading the weights marks the node fitted. There are no `TRAINABLE_BUFFERS`.

### Phase 1 fit

`statistical_initialization` is the whole training: one pass over the normal stream accumulates
per-band mean/variance (float64 Welford merge) while collecting raw `pool_size`-averaged spectra on
the `bank_stride` grid; the samples are standardised afterwards (pooling commutes with the per-band
affine map), capped to `max_bank_size` rows and reduced to `coreset_size` rows by
`cuvis_ai_patchcore.sampling.k_center_greedy` — the anomalib `KCenterGreedy` selection order on
exact distances (its sparse random projection is optional: `coreset_projection: sparse_random`) and
seeded for reproducibility. Run it with `StatisticalTrainer` or `restore-trainrun`; see
[`examples/trainrun_patchcore_cu3s.yaml`](examples/trainrun_patchcore_cu3s.yaml).

### Memory bank on deep features

The same node scores a dense feature grid — e.g. the `[B, 24, 24, 768]` patch tokens of a ViT —
when the features are used as they are: `input_channels: 768`, `standardize: false`, `pool_size: 1`,
`stride: 1`, `bank_stride: 1`, and the cube connected to `reference` so the 24×24 distance map is
upsampled to the cube resolution. Two banks (raw spectra, deep features) averaged with
`ScoreMapFusion` see complementary anomalies: spectral outliers and shape / texture novelty.

### Latency knobs

The scoring cost is `(H/stride)·(W/stride)·coreset_size·C` multiply-adds. Raising `stride` and
lowering `coreset_size` trade spatial resolution and bank coverage for speed; `autocast_dtype:
float16` roughly halves the distance-matrix time on CUDA. Validate any change on held-out data —
the defaults are the configuration validated for a 61-band VNIR cube at ~1000×1000 px.

## ScoreRangeNormalizer

`cuvis_ai_patchcore.node.calibration.ScoreRangeNormalizer` — put one detector's map on its normal
scale before fusing. Phase 1 pools the (subsampled, capped) scores of the normal frames and stores
their `low` / `high` percentiles (default 1 / 99) in the `lo` / `hi` buffers; inference maps them to
0 / 1 with `(x - lo) / (hi - lo)`, floored at 0 (`floor: true`) and **never clamped above**, so an
anomaly scoring far beyond the normal range keeps its rank. Two alternatives were measured to break
the two-bank fusion: a min-max range (set by the single most extreme normal pixel) and a clamped
percentile range (saturates every anomaly on a drifted session).

| Port | Direction | Shape / dtype |
|---|---|---|
| `scores` | in | `[B, H, W, C]` float32, `C == n_channels` (1 for an anomaly map) |
| `normalized` | out | `[B, H, W, C]` float32, `>= 0`, unbounded above |

hparams: `n_channels` 1 · `low` 1.0 · `high` 99.0 · `floor` true · `fit_subsample` 4 (spatial stride
of the Phase-1 collection) · `max_fit_values` 4 000 000 (seeded cap) · `seed` 0 · `eps` 1e-9.

## ScoreMapFusion

`cuvis_ai_patchcore.node.fusion.ScoreMapFusion` — fuse N score maps into one.

| Port | Direction | Shape / dtype | Notes |
|---|---|---|---|
| `scores` | in, variadic | `[B, H, W, 1]` float32 | one inbound connection per map; all maps share one shape |
| `scores` | out | `[B, H, W, 1]` float32 | fused map |

`mode`: `mean` (default) · `min` (AND) · `max` (OR) · `wmean` with `weights` (one per map,
normalised to sum to one) · `first`: per frame, the first inbound map (in connection order) that is
not all zero. `first` is for gated maps: connect the preferred detector's gated map first, and a
second detector's map shows only on frames the first gate blanks. Feed the other modes maps on a
common scale — a fitted normalizer per detector — otherwise the detector with the widest range
dominates. Stateless and differentiable.

A complete two-bank pipeline (raw-spectra bank + SteerViT-feature bank, each calibrated by
`ScoreRangeNormalizer`, fused by `ScoreMapFusion`) and its Phase-1 trainrun ship with
[cuvis-ai-steervit](https://github.com/cubert-hyperspectral/cuvis-ai-steervit) under `examples/`,
since that side needs both plugins.

## FrameScoreGate

`cuvis_ai_patchcore.node.gate.FrameScoreGate` — show an anomaly map only on anomalous frames. The
frame score is the mean of the top `topk_frac` pixels (default 0.1 %) of the alarm map, by the same
rule as `PatchCoreDetector.anomaly_score`, so a gate on a detector map reproduces the detector's
score. At or below `threshold` the frame's output is all zeros. `alarm_scores` (optional) lets the
gate score one map while it displays another, e.g. alarm on a robust feature bank, display a sharper
fusion map.

| Port | Direction | Shape / dtype | Notes |
|---|---|---|---|
| `scores` | in | `[B, H, W, C]` float32 | the display map |
| `alarm_scores` | in, optional | `[B, H, W, C]` float32 | map that drives the gate; default `scores` |
| `scores` | out | `[B, H, W, C]` float32 | display map (or its `mask_threshold` binary mask) on passing frames, zeros otherwise |
| `frame_score` | out | `[B]` float32 | raw per-frame alarm score |
| `passed` | out | `[B]` int32 | 1 when the (smoothed) score > `threshold` |

hparams: `threshold` (required; set above the session's clean band) · `topk_frac` 0.001 · `mode`
`heatmap` | `mask` · `mask_threshold` (default `threshold`) · `log_scores` false (log every frame
decision at INFO, for calibrating on a live session) · `smooth_k` 1 (> 1: gate on the rolling median
of the last k frame scores; runtime state, one frame per forward). Stateless otherwise: the
threshold is a hyper-parameter, not a fitted buffer, because the operating point drifts with the
session.

Two gates fused by `ScoreMapFusion(mode="first")` make an OR alarm with a priority display: each
gate alarms on its own detector, and the output shows the first detector's map whenever its gate
opens, the second detector's map only on frames the first gate misses.

## Install

One manifest file is one plugin. For development, point it at a checkout (the path is relative to
the manifest file):

```yaml
name: patchcore
path: "../cuvis-ai-patchcore"
package_name: cuvis-ai-patchcore
capabilities:
  - class_name: cuvis_ai_patchcore.node.patchcore.PatchCoreDetector
  - class_name: cuvis_ai_patchcore.node.calibration.ScoreRangeNormalizer
  - class_name: cuvis_ai_patchcore.node.fusion.ScoreMapFusion
  - class_name: cuvis_ai_patchcore.node.gate.FrameScoreGate
```

For a frozen, reproducible install, pin a release tag instead:

```yaml
name: patchcore
repo: "https://github.com/cubert-hyperspectral/cuvis-ai-patchcore.git"
tag: "v0.1.0"
package_name: cuvis-ai-patchcore
capabilities:
  - class_name: cuvis_ai_patchcore.node.patchcore.PatchCoreDetector
  - class_name: cuvis_ai_patchcore.node.calibration.ScoreRangeNormalizer
  - class_name: cuvis_ai_patchcore.node.fusion.ScoreMapFusion
  - class_name: cuvis_ai_patchcore.node.gate.FrameScoreGate
```

[`plugins.yaml`](plugins.yaml) is the local-path manifest of this repository, with the palette
metadata (category, tags, icon, port specs) generated by cuvis-ai's `emit_metadata`. Pipelines
reference the plugin by its name in their `plugins:` list:

```yaml
plugins:
  - cuvis_ai_builtin
  - patchcore
```

A minimal pipeline is in [`examples/patchcore_cu3s.yaml`](examples/patchcore_cu3s.yaml).

## Development

CI runs the suite on Python 3.11 and 3.13 with the committed lock:

```bash
uv run --no-sources --locked --extra dev pytest tests/ -m "not slow"
uv run --no-sources --locked --extra dev ruff format --check cuvis_ai_patchcore tests
uv run --no-sources --locked --extra dev ruff check cuvis_ai_patchcore tests
uv run python -c "from cuvis_ai_core.utils.node_registry import NodeRegistry; r=NodeRegistry(); r.register_plugin('plugins.yaml'); print(r.list_plugins())"
```

A plain `uv sync` installs the `cuda` dependency group: torch and torchvision from the PyTorch cu128
index (cu130 on aarch64 Linux, e.g. Jetson). The pins are scoped to that group, so an environment
that installs the plugin as a path or git dependency inherits none of them. After a dependency
change, regenerate the lock with `uv lock --no-sources` (CI resolves torch from PyPI).

## References

- Roth, K. et al. *Towards Total Recall in Industrial Anomaly Detection.* CVPR 2022 (PatchCore).
- Sener, O. & Savarese, S. *Active Learning for Convolutional Neural Networks: A Core-Set Approach.*
  ICLR 2018 (k-center greedy).

## License

Apache-2.0 — see [LICENSE](LICENSE).
