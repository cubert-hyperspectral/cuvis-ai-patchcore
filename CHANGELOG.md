# Changelog

## [Unreleased]

## 0.3.0 - 2026-10-07

### Added
- Added a boolean `decisions` output to `FrameScoreGate`: the pixels of the display map above
  `mask_threshold` (default `threshold`) on passing frames, all False on the others, as an object
  mask a viewer can overlay. `mask_threshold` now also sets this mask in `heatmap` mode; the other
  outputs are unchanged. On the walnut FO pipelines, a `mask_threshold` at the highest pixel of the
  session's clean frames gives a mask whose area follows the object (fake shells IoU 0.50-0.74,
  against 0.08-0.13 for a fixed top-0.5 % mask) and that stays empty on clean frames.
- Added `DecisionFusion`: combines N boolean masks by `any`, `all` or `first` (per frame the first
  mask with a set pixel, the mask of the map a `ScoreMapFusion(mode="first")` displays).
- Added `MaskComposite`: merges N boolean masks into one label map (`mask`, int32) and one level
  map (`scores`), e.g. shells = 1 and foreign objects = 2 in one view; where masks overlap the largest
  label / level wins.
- Added mode `softmin` to `ScoreMapFusion`: a soft minimum `-(1/beta) log(sum_i w_i exp(-beta x_i))`
  with the sharpness `beta` (required) and optional weights, between the hard minimum (large `beta`)
  and the mean (small `beta`). On the walnut FO data a soft minimum of the multi-scale SteerViT map
  and a spectral mixture map keeps the objects both see and drops each one's private false alarms.
- Added `invert` to `ScoreRangeNormalizer`: calibrates the negated map, for maps where higher means
  more normal (a log-likelihood). The default (`False`) is unchanged.
- Added `GridSubsample` (every `stride`-th pixel of a cube in both axes) and `ScoreUpsample` (a score
  map resized to the height and width of a reference tensor, bilinear by default), to score a
  per-pixel model on a coarse grid and bring its map back to full resolution.
- Added `ScoreMapSuppression`: a score map scaled by `1 - weight` inside a boolean mask shrunk by
  `erode_px` (defaults 1.0 / 4), e.g. an FO anomaly map inside a segmenter's shell mask, so that
  the detector cannot alarm on walnut shells smeared by motion.
- Added `MaskPersistence`: keeps a pixel of a boolean mask only if the previous frame's mask has a
  set pixel within `radius_px` (default 40, square window); the first frame shows nothing. Behind
  the walnut FO gate, real objects on the turntable stay in view while one-frame false blobs go
  (labelled 1-Oct frames: false blobs per FO frame 0.47 -> 0.25 with the deployed banks, 0.17 ->
  0.07 with the refit; each FO material shows on 1-4 fewer frames, mostly an object's first frame,
  stems on 4-7 fewer). Runtime state only, nothing serialized.
- Added `MaskMinArea`: drops the 8-connected blobs of a boolean mask below `min_area` pixels
  (default 250). On the user-labelled walnut frames it halves the false FO blobs per FO frame
  (0.475 -> 0.215 for the deployed banks, 0.170 -> 0.060 for the refit) and loses no FO object
  (one stem frame of 73 for the refit); specks on the empty belt are 42 px (median), FO blobs 9050 px.
- Added `ScoreMapSmoothing`: a score map convolved with a Gaussian of `sigma_px` (default 8,
  separable, mirrored border), PatchCore's own post-processing, before the gate. With the 1-Oct
  refit banks on the user's labels (sigma 8 + MaskMinArea 250): fake shells shown 106 -> 154 of
  182, stems 60 -> 69 of 73, false blobs per FO frame 0.170 -> 0.090.
- Added `SpectralObjectMask` (the pixels whose spectral angle to the frame's median spectrum
  exceeds `min_angle_deg`, default 6, on a stride-4 grid) and `MaskBlobGate` (the blobs of a
  mask that hold at least `min_px` pixels of a second mask): together a class-agnostic object
  gate for an anomaly mask. On the user-labelled walnut frames no FO object is lost and the false
  blobs per FO frame drop 0.475 -> 0.305 (deployed banks; 0.170 with MaskMinArea 250); on a
  recording with a wrong white reference the marks on the empty belt drop from 10.6 per frame to 0.
- Added `MaskBlobFilter`: `MaskMinArea`, `SpectralObjectMask` and `MaskBlobGate` in one pass (one
  labelling, both tests per blob, nothing done on an empty mask, the spectral angle on the cell
  grid without a GPU sync), identical to the chain. Behind the walnut FO gate with
  `MaskPersistence`: +1.5 to +2.7 ms per FO frame on an RTX 4070 laptop GPU, ~0 on clean frames;
  the four walnut_final_robust pipelines match an independent reimplementation on 32 real frames.

- Added `MaskPeakGate`: keeps the blobs of a mask whose peak score reaches `ratio` (default 0.8)
  times the peak of the reference blob they lie in (cell grid, lazy on empty masks). With
  `SpectralObjectMask(threshold="otsu", fill=True, dilate_px=4)`, `DecisionFusion("all")` and
  `MaskMinArea(100)` it cuts an anomaly mask to the objects (walnut FO: about 88 % less marked area
  off FOs and shells, no FO lost on the labelled frames, 4 of 240 on a fast turntable, about +2 ms
  per frame with marks).
- Added `threshold="otsu"` (each frame's Otsu level of the angle map, clipped to
  `otsu_floor_deg` / `otsu_ceiling_deg`), `fill` (closing + hole filling on the stride grid) and
  `dilate_px` (the full-size mask grown) to `SpectralObjectMask`; defaults unchanged.
- Added `core_ratio` (default 1.0) and the output `core` to `FrameScoreGate`: the pixels above
  `core_ratio` x `mask_threshold` on passing frames, the confident core of the mask, to fuse back in
  after a cut of the mask (it follows `mask_threshold` when that is recalibrated). With the walnut
  FO cut at 1.3 the loose stems of the 2 Oct production recording keep their marks (marks removed
  that the uncut mask shows: 74 -> 8, none on a stem).
- Added `invert` to `MaskBlobGate` (default `False`, unchanged): keeps the other blobs, those with
  fewer than `min_px` pixels of the gating mask. With `min_px=1`, a mask before a cut as
  `decisions` and the mask after it as `mask`, these are the marks the cut removed entirely; fused
  back with `DecisionFusion("any")`, the cut trims marks but never deletes one whose cells touch no
  kept piece.
- `FrameScoreGate.reset()` forgets the `smooth_k` history (e.g. a new recording); a batch of
  several frames with `smooth_k > 1` raises instead of sharing one rolling window.

### Changed
- `SpectralObjectMask` computes the angle with a matrix-vector product on the strided grid (no
  copy) and grows by `dilate_px` on the stride grid when that is exact; `MaskMinArea` and
  `MaskBlobGate` make one host copy per call.
- `MaskMinArea` and `MaskBlobGate` label the blobs on a grid of `cell` x `cell` pixel cells
  (new hparam `cell`, default 4; `cell=1` labels every pixel): exact pixel counts, marks whose
  cells touch form one blob. 28 ms -> ~1 ms per walnut frame for the whole robust mask.
- `SpectralObjectMask` takes the median over every `median_stride`-th pixel (new hparam, default 8)
  along the contiguous axis (4x faster on a GPU).
- `MaskPersistence` returns at once when this frame or the one before is empty, keeps the
  previous frame's emptiness (one GPU sync per frame instead of two) and sums the integral image
  in int32 (2x faster than the int64 default).
- `FrameScoreGate(log_scores=True)` also logs `pmax`, the display map's highest pixel per frame (the
  map `mask_threshold` cuts), so a live session's log alone is enough to set both thresholds from
  clean frames.

## 0.2.0 - 2026-09-28

### Added
- Added `tf32` to `PatchCoreDetector`: TF32 tensor-core matmuls in the float32 nearest-neighbour
  search (float32 storage and accumulation), set around the search and restored afterwards;
  ignored under `autocast_dtype`. On Jetson Thor it halves the 48 x 48 feature bank
  (9.6 -> 4.5 ms).

### Fixed
- Fixed the release workflow uploading uv's `dist/.gitignore` as a release asset
  (`default.gitignore` on v0.1.1): it now uploads the wheel and the sdist only.

## 0.1.1 - 2026-09-28

### Fixed
- Fixed every dependency resolution that reads the `cuda` group's index pins (`uv sync`,
  `uv run`, the release workflow), which failed with "conflicting indexes for package torch": the
  base torch requirement is declared once per index fork, as in cuvis-ai. The v0.1.0 release
  workflow failed on this; installing the v0.1.0 tag as a git or path dependency is not affected.

### Added
- Added a CI step that resolves the project with its index sources (`uv lock --dry-run`) and a
  guard test that the base requirements follow the fork markers.

## 0.1.0 - 2026-09-28

### Added
- Added `PatchCoreDetector`, HSI-PatchCore: per-band z-scored spectra, `pool_size`×`pool_size`
  local averaging, stride-grid sampling, a k-center-greedy coreset memory bank fitted in Phase 1
  (`statistical_initialization`), nearest-coreset Euclidean distance scoring and bilinear
  upsampling to the cube resolution. Emits `scores [B, H, W, 1]` and `anomaly_score [B]`, the mean
  of the top `max(1, floor(topk_frac * H * W))` pixel scores.
- Added the feature-grid mode of `PatchCoreDetector`: `standardize=False` scores any dense
  `[B, H, W, C]` feature grid (e.g. ViT patch tokens on their patch grid) with its raw values, the
  `mu` / `sd` buffers staying at identity; the optional `reference` input sets the resolution of
  `scores`, so a coarse feature-grid map is upsampled to the cube resolution.
- Added `autocast_dtype` (`float16` / `bfloat16`) for reduced-precision nearest-neighbour search on
  CUDA. The features are pre-scaled by 1/16 and, when standardizing, clamped to ±64 σ, so a far
  out-of-distribution pixel cannot overflow float16. `stride` / `coreset_size` / `chunk_size` set
  the latency–accuracy trade-off.
- Added anomalib's coreset recipe as an option: `coreset_projection="sparse_random"` /
  `projection_eps` select the coreset on a very sparse random projection to the
  Johnson-Lindenstrauss dimension (`density = 1/sqrt(F)`); scoring always uses the original
  features. Default off (exact distances). Seeded, so unlike anomalib's global-RNG selection a fit
  is reproducible.
- Added `cuvis_ai_patchcore.sampling`: `k_center_greedy`, dependency-free farthest-first coreset
  selection with anomalib `KCenterGreedy` semantics, and `sparse_random_projection`.
- Added `ScoreRangeNormalizer`: Phase-1 calibration of a score map onto its normal range, the
  pooled `low` / `high` percentiles (1 / 99) of subsampled normal scores mapped to 0 / 1, floored
  at 0 and never clamped above, so an anomaly scoring beyond the normal range keeps its rank. It is
  the calibration a multi-bank fusion needs: a min-max range is set by one extreme pixel, and a
  clamped percentile range saturates on drifted sessions.
- Added `ScoreMapFusion`: variadic fan-in fusion of N score maps by mean / min / max / weighted
  mean, or `first` (per frame, the first inbound map that is not all zero: a priority display of
  gated maps).
- Added `FrameScoreGate`: a stateless post-processor that zeros an anomaly map unless its
  top-`topk_frac` frame score exceeds `threshold` (`mode: heatmap` passes the map through, `mask`
  binarises it at `mask_threshold`); emits the gated `scores [B, H, W, C]`, `frame_score [B]` and
  `passed [B]`. The frame score follows the detector's top-k rule, so a gate on a detector map
  reproduces its `anomaly_score`. The optional `alarm_scores` input alarms on one map while another
  is displayed. Live-calibration helpers: `log_scores` logs every frame decision, `smooth_k` gates
  on the rolling median of the last k frame scores. Generic node, planned to move to cuvis-ai core.
- Added the local-path plugin manifest (`plugins.yaml`) with generated palette metadata, an
  example cu3s pipeline and Phase-1 trainrun (`examples/`), and tests: golden parity against the
  reference scoring formula, port contracts, fit statistics, coreset membership, state-dict
  round-trips, manifest loading and pipeline reload smokes (single bank, two-bank fusion,
  calibrated fusion, and gates feeding `ScoreMapFusion(mode="first")`).
- Added the `cuda` dependency group for local GPU development: torch and torchvision come from the
  cu128 index (cu130 on aarch64 Linux / Jetson). The pins are scoped to the group, so an
  environment that installs the plugin as a path or git dependency inherits none; a guard test
  checks this and that the committed lock (the CI lock) resolves torch from PyPI.
- Targets cuvis-ai-core >= 0.17.4 and cuvis-ai-schemas >= 0.12.0 on Python 3.11 – 3.13.
- Added CI on Python 3.11 and 3.13 for every pull request (stacked PRs included) and the weekly
  dependency compatibility audit against cuvis-ai-core v0.17.4.
