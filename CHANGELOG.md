# Changelog

## [Unreleased]

### Added
- Added `FrameScoreGate`: a stateless post-processor that zeros an anomaly map unless its
  top-`topk_frac` frame score exceeds `threshold` (`mode: heatmap` passes the map through, `mask`
  binarises it at `mask_threshold`); emits the gated `scores [B, H, W, C]`, `frame_score [B]` and
  `passed [B]`. The optional `alarm_scores` input alarms on one map while another is displayed.
  Live-calibration helpers: `log_scores` logs every frame decision, `smooth_k` gates on the rolling
  median of the last k frame scores. Generic node, planned to move to cuvis-ai core.
- Added `PatchCoreDetector` node — HSI-PatchCore: per-band z-scored spectra, `pool_size`×`pool_size`
  local averaging, stride-grid sampling, k-center-greedy coreset memory bank fitted in Phase 1
  (`statistical_initialization`), nearest-coreset Euclidean distance scoring, bilinear upsampling
  to cube resolution. Emits `scores [B,H,W,1]` and `anomaly_score [B]` (top-k mean).
- Added `autocast_dtype` (`float16` / `bfloat16`) for reduced-precision nearest-neighbour search
  on CUDA, plus `stride` / `coreset_size` / `chunk_size` knobs for the latency–accuracy trade-off.
- Added `cuvis_ai_patchcore.sampling.k_center_greedy` — dependency-free farthest-first coreset
  selection (anomalib `KCenterGreedy` semantics without the sparse random projection, which is
  unnecessary for low-dimensional spectral features).
- Added the local-path plugin manifest (`plugins.yaml`), an example cu3s pipeline + Phase-1 trainrun
  (`examples/`), and tests: golden parity against the reference scoring formula, port contract,
  fit statistics, coreset membership, state-dict round-trip, manifest loading, pipeline reload smoke.
- Added feature-grid mode to `PatchCoreDetector`: `standardize=False` scores any dense
  `[B, H, W, C]` feature grid (e.g. ViT patch tokens on their patch grid) with its raw values; the
  `mu` / `sd` buffers stay at identity, so the state-dict layout is unchanged.
- Added the optional `reference` input to `PatchCoreDetector`: its spatial size sets the resolution
  of `scores`, so a coarse feature-grid map is upsampled to the cube resolution.
- Added `ScoreMapFusion`: variadic fan-in fusion of N score maps by mean / min / max / weighted mean,
  or `first` (per frame, the first inbound map that is not all zero: a priority display of gated maps).
- Added `ScoreRangeNormalizer`: Phase-1 calibration of a score map onto its normal range, the
  pooled `low` / `high` percentiles (1 / 99) of subsampled normal scores mapped to 0 / 1, floored
  at 0 and never clamped above, so an anomaly scoring beyond the normal range keeps its rank. The
  calibration the two-bank fusion needs (min-max is set by one extreme pixel, a clamped percentile
  range saturates on drifted sessions).
- Added anomalib's coreset recipe as an option: `cuvis_ai_patchcore.sampling.sparse_random_projection`
  (very sparse random projection to the Johnson-Lindenstrauss dimension, `density = 1/sqrt(F)`) and
  `k_center_greedy(..., projection_eps=...)`, exposed on `PatchCoreDetector` as
  `coreset_projection="sparse_random"` / `projection_eps` (default off: exact distances). Seeded, so
  unlike anomalib's global-RNG selection a fit is reproducible.
- Added the `cuda` dependency group for local GPU development: torch and torchvision come from the
  cu128 index (cu130 on aarch64 Linux / Jetson). The pins are scoped to the group, so an environment
  that installs the plugin as a path or git dependency inherits none; a guard test checks this and
  that the committed lock (the CI lock) resolves torch from PyPI.
- Targets cuvis-ai-core >= 0.17.4 and cuvis-ai-schemas >= 0.12.0 on Python 3.11-3.13.
- Added CI on Python 3.11 and 3.13 for every pull request (stacked PRs included) and the weekly
  dependency compatibility audit against cuvis-ai-core v0.17.4.

### Changed
- Reduced-precision guard: the +-64 clamp applies only when standardizing (it is a z-unit
  assumption); raw feature grids get the 1/16 pre-scaling alone.
- `PatchCoreDetector.anomaly_score` and `FrameScoreGate.frame_score` share one top-k rule,
  `k = max(1, floor(topk_frac * n))` over the n values of a frame, rounded once: the detector's
  two-step product floored one pixel lower for a few map sizes (e.g. 1160 x 25), so a gate on a
  detector map now reproduces the detector's score exactly. Sizes such as 1000 x 1080 are unchanged.
- The nodes share one private helper module for the top-k score, the `topk_frac` check, the seeded
  Phase-1 row cap and the unfitted-`forward` guard; `ScoreRangeNormalizer` thins its Phase-1 values
  with cuvis-ai-core's `subsample_hw`. Outputs and fitted buffers are bit-identical.
