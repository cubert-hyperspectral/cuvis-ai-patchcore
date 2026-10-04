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

The plugin ships fourteen nodes:

| Node | Role |
|---|---|
| [`PatchCoreDetector`](#patchcoredetector) | memory-bank detector on spectra or any dense feature grid (Phase-1 fitted) |
| [`ScoreRangeNormalizer`](#scorerangenormalizer) | puts one detector's map on its normal range before fusion (Phase-1 fitted) |
| [`ScoreMapFusion`](#scoremapfusion) | fuses N maps: mean, min, max, weighted mean, soft minimum, or a priority rule for gated maps |
| [`FrameScoreGate`](#framescoregate) | blanks a map on frames whose top-k score stays at or below a threshold; emits the object mask |
| [`DecisionFusion`](#decisionfusion) | fuses N boolean masks: any, all, or the priority rule of `ScoreMapFusion` |
| [`MaskComposite`](#maskcomposite) | merges N boolean masks into one label map and one level map for display |
| [`GridSubsample`](#gridsubsample-and-scoreupsample) | every `stride`-th pixel of a cube, to score a per-pixel model on a coarse grid |
| [`ScoreUpsample`](#gridsubsample-and-scoreupsample) | resizes a grid's score map to the height and width of a reference tensor |
| [`ScoreMapSuppression`](#scoremapsuppression) | down-weights a score map inside a boolean mask shrunk by a margin (e.g. a segmenter's mask of objects that cannot be anomalous) |
| [`MaskPersistence`](#maskpersistence) | keeps a mask pixel only where the previous frame's mask lies within a radius, so one-frame flickers never show |
| [`MaskMinArea`](#maskminarea) | drops the blobs of a mask below a pixel count, e.g. specks on an empty background |
| [`ScoreMapSmoothing`](#scoremapsmoothing) | Gaussian smoothing of a score map (PatchCore's post-processing) |
| [`SpectralObjectMask`](#spectralobjectmask-and-maskblobgate) | marks the pixels that are not the background material (spectral angle to the frame's median) |
| [`MaskBlobGate`](#spectralobjectmask-and-maskblobgate) | keeps the blobs of a mask that hold enough pixels of a second mask (e.g. objects) |
| [`MaskBlobFilter`](#maskblobfilter) | MaskMinArea + SpectralObjectMask + MaskBlobGate in one lazy pass, for a live pipeline |
| [`MaskPeakGate`](#maskpeakgate-and-the-pixel-level-cut) | keeps the blobs of a mask whose peak score reaches a share of the peak of their reference blob |

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
of the Phase-1 collection) · `max_fit_values` 4 000 000 (seeded cap) · `seed` 0 · `eps` 1e-9 ·
`invert` false. With `invert: true` the node calibrates `-x` (fit and inference), for maps where
higher means more normal, e.g. a Gaussian mixture's log-likelihood; the output is then an anomaly
score on the same normal-range scale as the other detectors.

## ScoreMapFusion

`cuvis_ai_patchcore.node.fusion.ScoreMapFusion` — fuse N score maps into one.

| Port | Direction | Shape / dtype | Notes |
|---|---|---|---|
| `scores` | in, variadic | `[B, H, W, 1]` float32 | one inbound connection per map; all maps share one shape |
| `scores` | out | `[B, H, W, 1]` float32 | fused map |

`mode`: `mean` (default) · `min` (AND) · `max` (OR) · `wmean` with `weights` (one per map,
normalised to sum to one) · `softmin` with `beta` (required) and optional `weights`: the soft minimum
`-(1/beta) log(sum_i w_i exp(-beta x_i))`, a soft AND between the minimum (large `beta`; at most
`log(N) / beta` above it with equal weights) and the mean (small `beta`) · `first`: per frame, the
first inbound map (in connection order) that is not all zero. `first` is for gated maps: connect the
preferred detector's gated map first, and a second detector's map shows only on frames the first
gate blanks. Feed the other modes maps on a common scale — a fitted normalizer per detector —
otherwise the detector with the widest range dominates. Stateless and differentiable.

Order-dependent rules (`first`, weighted `wmean` / `softmin`): `save_to_file` writes a fan-in's
connections in the order their source nodes entered the pipeline graph, and a reloaded pipeline uses
that order. Add the source nodes in the intended order (or check the saved yaml).

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
| `decisions` | out | `[B, H, W, C]` bool | object mask: display-map pixels above `mask_threshold` on passing frames, all False otherwise |

hparams: `threshold` (required; set above the session's clean band) · `topk_frac` 0.001 · `mode`
`heatmap` | `mask` · `mask_threshold` (default `threshold`; the cutoff of `decisions`, and of
`scores` in `mask` mode) · `log_scores` false (log every frame decision and the display map's
highest pixel `pmax` at INFO, for calibrating both thresholds on a live session) · `smooth_k` 1 (> 1: gate on the rolling median of the last k frame scores; runtime
state, one frame per forward). Stateless otherwise: the thresholds are hyper-parameters, not fitted
buffers, because the operating point drifts with the session.

**Object mask.** Set `mask_threshold` on the display map's own scale, e.g. to the highest pixel of
the session's clean frames. The mask then marks the pixels above anything a clean frame produced:
its area follows the object, and it stays empty on clean frames even if the gate is bypassed. A
per-frame quantile (the top q of the pixels) marks the same area on every frame instead, too small
for a large object and spread over texture on a small one. `decisions` is a port name that viewers
such as cuvis.next overlay as a mask.

Two gates fused by `ScoreMapFusion(mode="first")` make an OR alarm with a priority display: each
gate alarms on its own detector, and the output shows the first detector's map whenever its gate
opens, the second detector's map only on frames the first gate misses. Their `decisions` fused by
`DecisionFusion(mode="first")` give the mask of the displayed map.

## DecisionFusion

`cuvis_ai_patchcore.node.fusion.DecisionFusion` — fuse N boolean masks into one.

| Port | Direction | Shape / dtype | Notes |
|---|---|---|---|
| `decisions` | in, variadic | `[B, H, W, C]` bool | one inbound connection per mask; all masks share one shape |
| `decisions` | out | `[B, H, W, C]` bool | fused mask |

`mode`: `any` (default, pixel-wise OR) · `all` (pixel-wise AND) · `first`: per frame, the first
inbound mask (in connection order) with a set pixel, all False when none has one. Stateless.

## MaskComposite

`cuvis_ai_patchcore.node.fusion.MaskComposite` — merge N boolean masks into one output, e.g. the
shells of a segmentation and the object mask of an anomaly gate in one view.

| Port | Direction | Shape / dtype | Notes |
|---|---|---|---|
| `decisions` | in, variadic | `[B, H, W, C]` bool | one inbound connection per mask, in the order of `labels` / `levels`; one B, H, W |
| `mask` | out | `[B, H, W]` int32 | label map: 0 where no mask is set, else the largest label among the masks set there |
| `scores` | out | `[B, H, W, 1]` float32 | level map: 0 where no mask is set, else the largest level among the masks set there |

`labels` (default `[1, 2]`, integers >= 1) and `levels` (default `[0.5, 1.0]`, finite, >= 0): one
per mask. Where masks overlap the largest entry wins, so the mask that should stay on top gets the
largest one. A mask with several channels counts where any channel is set. Stateless.

Viewers show a `mask` port as a label mask and a `scores` port as a heatmap. A port another node
consumes is not a terminal output any more, so a pipeline that also wants the single masks shown
adds a copy of them, e.g. `DecisionFusion` over one mask.

## GridSubsample and ScoreUpsample

`cuvis_ai_patchcore.node.spatial.GridSubsample` / `ScoreUpsample` — score a per-pixel model on a
coarse grid of the cube and bring its map back to full resolution. A per-pixel spectral model costs
the same for every pixel, so on a 1000 x 1080 cube a stride-4 grid is 16 x cheaper.

| Node | Port | Direction | Shape / dtype |
|---|---|---|---|
| `GridSubsample` | `cube` | in | `[B, H, W, C]` float32 |
| | `cube` | out | `[B, ceil(H / stride), ceil(W / stride), C]` = `cube[:, ::stride, ::stride, :]` |
| `ScoreUpsample` | `scores` | in | `[B, h, w, C]` float32 (the grid's map) |
| | `reference` | in | any `[B, H, W, *]` float32 of the target size (e.g. the cube) |
| | `scores` | out | `[B, H, W, C]` float32, `align_corners=False` |

hparams: `GridSubsample` — `stride` 4 · `ScoreUpsample` — `mode` `bilinear` (or `bicubic`,
`nearest`). Both are stateless and differentiable. A spectral branch, e.g. the builtin
`SNVCorrection` and `GaussianMixtureClusterer` on the grid, then `ScoreUpsample`, then
`ScoreRangeNormalizer(invert=true)` on the log-likelihood, gives a map that `ScoreMapFusion`
(`softmin`) can fuse with an image model's map.

## ScoreMapSuppression

`cuvis_ai_patchcore.node.fusion.ScoreMapSuppression` — down-weight a score map inside a boolean
mask, e.g. an anomaly map inside a segmenter's mask of an object class that cannot be anomalous
(walnut shells), placed before the gate so that neither the frame score nor the object mask can
come from those objects.

| Port | Direction | Shape / dtype | Notes |
|---|---|---|---|
| `scores` | in | `[B, H, W, 1]` float32 | e.g. a fused anomaly map |
| `mask` | in | `[B, H, W, C]` bool | same B, H, W; a pixel counts where any channel is set |
| `scores` | out | `[B, H, W, 1]` float32 | `scores x (1 - weight x the eroded mask)` |

hparams: `weight` (default 1.0, in `[0, 1]`: the share of the score removed inside the mask) ·
`erode_px` (default 4, >= 0): the mask is first shrunk by a square erosion of this many pixels, so
an object touching a masked one keeps its score along the shared edge; the image border does not
shrink the mask. Stateless, differentiable in `scores`.

## MaskPersistence

`cuvis_ai_patchcore.node.temporal.MaskPersistence` — a frame-to-frame filter for the object mask
of a moving scene (a turntable, a belt): a pixel of the current mask is kept only if the previous
frame's mask has a set pixel within `radius_px` of it. A real object stays in view and moves a
bounded distance per frame, so it shows from its second frame on; a blob that lives for one frame
(sensor noise, motion blur) never shows. Placed after a `FrameScoreGate`'s `decisions`, before
the viewers of the mask.

| Port | Direction | Shape / dtype | Notes |
|---|---|---|---|
| `decisions` | in | `[B, H, W, C]` bool | consecutive frames, batch order = time order |
| `decisions` | out | `[B, H, W, C]` bool | the pixels with a previous-frame pixel (same channel) within `radius_px`; none on the first frame |

hparams: `radius_px` (default 40, >= 0): half the side of the square window, at least the
largest distance an object moves between two frames. Runtime state only (the last frame's input
mask, not serialized): the first frame after loading, after `reset()` or after a change of mask
shape or device shows nothing. Not differentiable (boolean masks).

## MaskMinArea

`cuvis_ai_patchcore.node.morphology.MaskMinArea` — drop the blobs of a boolean mask that have
fewer than `min_area` pixels. A real object's blob is the object plus the score map's halo; specks
from texture or noise on an empty background are a few dozen pixels. Placed after a
`FrameScoreGate`'s `decisions`, before the viewers of the mask (the alarm is unchanged).

The blobs are labelled on a grid of `cell` x `cell` pixel cells (default 4): the marked pixels per
cell are summed on the GPU and only the small cell grid is labelled (8-connected, OpenCV on the
CPU). Areas are exact pixel counts; the one difference to labelling every pixel is that marks
whose cells touch form one blob (gaps of up to `2 * cell - 1` pixels). `cell=1` labels every pixel.

| Port | Direction | Shape / dtype | Notes |
|---|---|---|---|
| `decisions` | in | `[B, H, W, C]` bool | each frame and channel on its own |
| `decisions` | out | `[B, H, W, C]` bool | the blobs with at least `min_area` pixels |

hparams: `min_area` (default 250, >= 0; `0` / `1` keep everything), `cell` (default 4, >= 1).
Stateless, not differentiable (boolean masks); an empty mask is returned without work.

## ScoreMapSmoothing

`cuvis_ai_patchcore.node.spatial.ScoreMapSmoothing` — convolve a score map with a normalised
Gaussian of `sigma_px` pixels (separable, radius round(4 sigma), mirrored border as OpenCV's
BORDER_REFLECT_101). Isolated one-patch peaks drop, regions several patches agree on keep
their level. Placed before a `FrameScoreGate`, so the alarm and the mask both see the smoothed
map (calibrate the gate on the smoothed map).

| Port | Direction | Shape / dtype | Notes |
|---|---|---|---|
| `scores` | in | `[B, H, W, C]` float32 | e.g. a fused anomaly map |
| `scores` | out | `[B, H, W, C]` float32 | each channel smoothed |

hparams: `sigma_px` (default 8.0, >= 0; `0` passes the map through). Stateless (the kernel is
not saved), torch-native, differentiable.

## SpectralObjectMask and MaskBlobGate

`cuvis_ai_patchcore.node.objectness.SpectralObjectMask` marks a pixel as an object where its
spectral angle to the frame's median spectrum (the background's, when it covers most of the
frame) exceeds `min_angle_deg` (default 6.0), computed on every `stride`-th pixel (default 4) and
expanded by nearest neighbour; the median runs over every `median_stride`-th pixel (default 8).
Brightness-invariant (shadows stay background) and relative to the same frame (a white-reference
error shifts background and objects alike); class-agnostic (any material that differs from the
background is an object, known or not). Outputs `decisions` `[B, H, W, 1]` bool and `angle`
`[B, H/s, W/s, 1]` float32. Options (defaults keep the fixed threshold):
- `threshold="otsu"`: each frame's Otsu level of the angle map (0.1 deg steps, OpenCV), clipped
  to `[otsu_floor_deg, otsu_ceiling_deg]` (default 3 / 12), instead of `min_angle_deg`;
- `fill=True`: closes 1-cell gaps (3 x 3) and fills the holes of the objects on the stride grid;
- `dilate_px`: grows the full-size mask by that many pixels (a margin around each object; on the
  stride grid when it is a multiple of `stride`, exact).

`cuvis_ai_patchcore.node.morphology.MaskBlobGate` keeps the blobs of `decisions` (labelled on the
cell grid of `MaskMinArea`) that hold at least `min_px` (default 16) pixels of `mask` (any
channel). With the object mask as `mask`, an anomaly blob on the empty background goes, one
around an object (halo included) stays. A foreign object with the background's own spectrum is
not an object to it.
With `invert=True` it keeps the other blobs (fewer than `min_px` pixels of `mask`): with `min_px=1`,
the mask before a cut as `decisions` and the mask after it as `mask`, the marks the cut removed
entirely, to fuse back in (`DecisionFusion("any")`).

## MaskPeakGate and the pixel-level cut

`cuvis_ai_patchcore.node.morphology.MaskPeakGate` keeps the blobs of `decisions` (on the cell grid
of `MaskMinArea`) whose highest `scores` value reaches `ratio` (default 0.8) times the highest score
of the `reference` blob they lie in; a blob outside every reference blob stays.

| Port | Direction | Shape / dtype | Notes |
|---|---|---|---|
| `decisions` | in | `[B, H, W, C]` bool | e.g. the pieces of an anomaly mask after a cut |
| `reference` | in | `[B, H, W, C']` bool | e.g. the mask before the cut (any channel counts) |
| `scores` | in | `[B, H, W, C'']` float32 | the map the mask was thresholded from (first channel), finite where marked |
| `decisions` | out | `[B, H, W, C]` bool | the blobs that reach `ratio` x their reference peak |

hparams: `ratio` (0.8, in [0, 1]; `0` keeps everything), `cell` (4). Stateless, not
differentiable; an empty mask is returned without work.

The pixel-level cut of an anomaly mask to the objects, four nodes after the gate (walnut FO,
`walnut_final_robust_v2/*_cut`):
1. `SpectralObjectMask(threshold="otsu", otsu_floor_deg=2, fill=True, dilate_px=4)`: the objects plus 4 px;
2. `DecisionFusion(mode="all")`: the mask AND the objects;
3. `MaskMinArea(min_area=100)`: leftover pieces under 100 px go;
4. `MaskPeakGate(ratio=0.8)` with the uncut mask as `reference` and the gate's input map as `scores`:
   halo pieces left on neighbouring objects, far below the mark's peak, go.
On the walnut FO stand this removes about 88 % of the marked area off the FOs and shells and nearly
all marks left on the empty belt (0.02-0.07 per frame remain), with no FO lost on the labelled 1-Oct
frames (4 of 240 on a fast turntable); about +2 ms per frame with marks on an RTX 4070 laptop GPU
(+1 ms without).
**Not FO-safe for dark, thin objects:** on a production recording with loose walnut stems (2 Oct)
the cut removes the stems' marks. A dark stem's spectrum differs from the belt's mostly in
brightness, which the spectral angle ignores (8-11 deg on a few cells, below the frame's Otsu
level), and a stem's mark that touches another kept mark is not brought back by the `invert` gate.

## MaskBlobFilter

`cuvis_ai_patchcore.node.objectness.MaskBlobFilter` — `MaskMinArea`, `SpectralObjectMask` and
`MaskBlobGate` in one pass, for a live pipeline: one labelling of the cell grid, both tests per
blob, nothing done on an empty mask, one small copy to the host. A blob is kept if it has at least
`min_area` marked pixels and at least `min_object_px` of them lie in object cells: cells whose
first pixel (the stride-`cell` grid) has a spectral angle above `min_angle_deg` to the per-band
median of every `median_stride`-th pixel. Identical to the three-node chain with the same
parameters (tested).

| Port | Direction | Shape / dtype | Notes |
|---|---|---|---|
| `decisions` | in | `[B, H, W, C]` bool | e.g. a `FrameScoreGate`'s decisions |
| `cube` | in, optional | `[B, H, W, K]` float32 | the frames' cube; without it only the size test runs |
| `decisions` | out | `[B, H, W, C]` bool | the blobs that pass both tests |

hparams: `min_area` (250), `min_object_px` (16; `0` skips the object test), `min_angle_deg` (6.0),
`cell` (4), `median_stride` (8). Stateless, not differentiable. On a 1000 x 1080 x 61 frame with
FO marks it takes about 1 ms on an RTX 4070 laptop GPU, 0.1 ms on a clean frame; with
`MaskPersistence` behind it the two walnut FO pipelines run +1.5 to +2.7 ms per FO frame.

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
  - class_name: cuvis_ai_patchcore.node.fusion.DecisionFusion
  - class_name: cuvis_ai_patchcore.node.fusion.MaskComposite
  - class_name: cuvis_ai_patchcore.node.spatial.GridSubsample
  - class_name: cuvis_ai_patchcore.node.spatial.ScoreUpsample
  - class_name: cuvis_ai_patchcore.node.fusion.ScoreMapSuppression
  - class_name: cuvis_ai_patchcore.node.temporal.MaskPersistence
  - class_name: cuvis_ai_patchcore.node.morphology.MaskMinArea
  - class_name: cuvis_ai_patchcore.node.spatial.ScoreMapSmoothing
  - class_name: cuvis_ai_patchcore.node.objectness.SpectralObjectMask
  - class_name: cuvis_ai_patchcore.node.morphology.MaskBlobGate
  - class_name: cuvis_ai_patchcore.node.objectness.MaskBlobFilter
  - class_name: cuvis_ai_patchcore.node.morphology.MaskPeakGate
```

For a frozen, reproducible install, pin a release tag instead (`GridSubsample`, `ScoreUpsample`,
`ScoreMapSuppression`, `MaskPersistence`, `MaskMinArea`, `ScoreMapSmoothing`, `SpectralObjectMask`,
`MaskBlobGate`, `MaskBlobFilter`, `MaskPeakGate`,
`ScoreMapFusion(softmin)` and
`ScoreRangeNormalizer(invert)` are not released yet, see the changelog):

```yaml
name: patchcore
repo: "https://github.com/cubert-hyperspectral/cuvis-ai-patchcore.git"
tag: "v0.3.0"
package_name: cuvis-ai-patchcore
capabilities:
  - class_name: cuvis_ai_patchcore.node.patchcore.PatchCoreDetector
  - class_name: cuvis_ai_patchcore.node.calibration.ScoreRangeNormalizer
  - class_name: cuvis_ai_patchcore.node.fusion.ScoreMapFusion
  - class_name: cuvis_ai_patchcore.node.gate.FrameScoreGate
  - class_name: cuvis_ai_patchcore.node.fusion.DecisionFusion
  - class_name: cuvis_ai_patchcore.node.fusion.MaskComposite
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
