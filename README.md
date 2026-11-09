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

The plugin ships one node, [`PatchCoreDetector`](#patchcoredetector): the memory-bank detector on
spectra or any dense feature grid (Phase-1 fitted). The score, mask and gate nodes it shipped up to
0.4.x are cuvis-ai built-ins; see [Moved to cuvis-ai](#moved-to-cuvis-ai).

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
| `search_backend` | `exact` | `exact` shortlists nearest-coreset candidates through `torch.cdist`'s matmul expansion (today's behaviour, unchanged); `turbovec` shortlists through an approximate cosine search over a quantised coreset copy; see [Search backend](#search-backend) |
| `search_bit_width` | 4 | turbovec quantisation width, `2` or `4`; only used with `search_backend: turbovec` |
| `shortlist_size` | 8 | candidate rows kept per query before rescoring, both backends; clipped to the number of coreset rows |
| `rescore` | `true` | recompute the exact fp32 distance over the shortlist; `false` reconstructs a distance from the approximate cosine and the exact row norms instead (only with `search_backend: turbovec`) |

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
upsampled to the cube resolution. Two banks (raw spectra, deep features) averaged with cuvis-ai's
`ScoreMapFusion` see complementary anomalies: spectral outliers and shape / texture novelty.

### Search backend

The nearest-coreset search always runs in two steps: a cheap shortlist of `shortlist_size`
candidate rows per query, then an exact fp32 distance over just those rows (`rescore: true`, the
default). The default `search_backend: exact` shortlists through `torch.cdist`; with
`shortlist_size: 8` this is byte-identical to every release before this backend existed.

`search_backend: turbovec` shortlists instead through an approximate cosine search over a
quantised, unit-normalised copy of the coreset, built on the CPU with the optional `turbovec`
dependency (`uv pip install cuvis-ai-patchcore[turbovec]`). Cosine ranking is **not** the same as
the Euclidean ranking PatchCore scores on: two coreset rows with different norms can rank
differently by angle than by distance, so a turbovec-configured node should be validated against
the `exact` backend on representative data before deployment: it is a candidate shortlisting
step, not a drop-in replacement, and its recall depends on how anisotropic the feature norms are.
`rescore: true` keeps the reported distance exact for every row that made the shortlist (only the
*candidate set* is approximate); `rescore: false` skips gathering the fp32 rows and instead
reconstructs a Euclidean distance from the approximate cosine and the coreset's exact norms,
trading some additional accuracy for lower cost.

The index is rebuilt automatically whenever the coreset's shape, dtype or content changes (a refit,
a weights load, a dtype cast), so it never needs manual invalidation, and it adds no keys to the
node's `state_dict`: a pipeline `.pt` saved with one backend loads unchanged into a node configured
for the other. There is no ONNX export path for `search_backend: turbovec` (the index runs outside
the traced graph); export a pipeline with `search_backend: exact` instead.

### Latency knobs

The scoring cost is `(H/stride)·(W/stride)·coreset_size·C` multiply-adds. Raising `stride` and
lowering `coreset_size` trade spatial resolution and bank coverage for speed; `autocast_dtype:
float16` roughly halves the distance-matrix time on CUDA. Validate any change on held-out data —
the defaults are the configuration validated for a 61-band VNIR cube at ~1000×1000 px.

## Moved to cuvis-ai

The fourteen score, mask and gate nodes of 0.2.x to 0.4.x are cuvis-ai built-ins since cuvis-ai
0.19.1 (one implementation, generic names), and 0.5.0 removed the plugin copies without a
deprecation release:

| up to 0.4.x | cuvis-ai built-in |
|---|---|
| `cuvis_ai_patchcore.node.calibration.ScoreRangeNormalizer` | `cuvis_ai.node.normalization.PercentileNormalizer` (`norm_mode: statistical`, `statistical_fit: pooled`, `quantile_low` / `quantile_high` = `low` / `high` divided by 100, `clamp: floor`, `invert`) |
| `cuvis_ai_patchcore.node.fusion.ScoreMapFusion` | `cuvis_ai.node.fusion.ScoreMapFusion` |
| `cuvis_ai_patchcore.node.fusion.DecisionFusion` | `cuvis_ai.node.fusion.DecisionFusion` |
| `cuvis_ai_patchcore.node.fusion.MaskComposite` | `cuvis_ai.node.mask_ops.MaskComposite` |
| `cuvis_ai_patchcore.node.fusion.ScoreMapSuppression` | `cuvis_ai.node.mask_ops.ScoreMapSuppression` |
| `cuvis_ai_patchcore.node.gate.FrameScoreGate` | `cuvis_ai.node.deciders.frame_score_gate.FrameScoreGate` |
| `cuvis_ai_patchcore.node.spatial.GridSubsample`, `ScoreUpsample`, `ScoreMapSmoothing` | `cuvis_ai.node.spatial.GridSubsample`, `ScoreUpsample`, `ScoreMapSmoothing` |
| `cuvis_ai_patchcore.node.morphology.MaskMinArea`, `MaskBlobGate`, `MaskPeakGate` | `cuvis_ai.node.mask_ops.MaskMinArea`, `MaskBlobGate`, `MaskPeakGate` |
| `cuvis_ai_patchcore.node.objectness.MaskBlobFilter` | `cuvis_ai.node.mask_ops.MaskBlobFilter` |
| `cuvis_ai_patchcore.node.objectness.SpectralObjectMask` | `cuvis_ai.node.spectral_angle_mapper.SpectralObjectMask` |
| `cuvis_ai_patchcore.node.temporal.MaskPersistence` | none: removed, as only superseded presets used it |

cuvis-ai's `migrate-plugin-nodes` rewrites a pipeline yaml and its `.pt` from the old classes to the
built-ins (bit for bit on the walnut presets). A pipeline that still names the old classes needs a
0.4.x release.

## Install

One manifest file is one plugin. For development, point it at a checkout (the path is relative to
the manifest file):

```yaml
name: patchcore
path: "../cuvis-ai-patchcore"
package_name: cuvis-ai-patchcore
capabilities:
  - class_name: cuvis_ai_patchcore.node.patchcore.PatchCoreDetector
```

For a frozen, reproducible install, pin a release tag instead:

```yaml
name: patchcore
repo: "https://github.com/cubert-hyperspectral/cuvis-ai-patchcore.git"
tag: "v0.5.0"
package_name: cuvis-ai-patchcore
capabilities:
  - class_name: cuvis_ai_patchcore.node.patchcore.PatchCoreDetector
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
