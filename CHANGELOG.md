# Changelog

## [Unreleased]

## 0.1.0 - 2026-09-25

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
