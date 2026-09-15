# Changelog

## [Unreleased]

### Added
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
