"""Coreset search backends for PatchCoreDetector.

``_nearest_distance`` splits into a per-backend shortlist step (candidate row ids, cheap and
approximate) and a shared fp32 rescoring step (the exact distance over just those rows). The
"exact" backend shortlists through ``torch.cdist``'s matmul expansion, unchanged from before this
module existed. The "turbovec" backend shortlists through an approximate cosine search over a
quantised, unit-normalised copy of the coreset: cosine ranking is not the same as the Euclidean
ranking PatchCore scores on when row norms differ, which is why ``rescore`` (the fp32 exact
distance over the shortlisted rows) exists, and why an unrescored run uses the exact row norms
to reconstruct a Euclidean distance from the approximate cosine instead of trusting the cosine
value as a distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

_BIT_WIDTHS = (2, 4)


def _pad_to_multiple_of_8(rows: Tensor) -> Tensor:
    """Zero-pad the channel dimension of ``[N, C]`` up to the next multiple of 8."""
    c = rows.shape[-1]
    pad = (-c) % 8
    if pad == 0:
        return rows
    return torch.nn.functional.pad(rows, (0, pad))


def _unit_normalize(rows: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    """Return ``(unit_rows, norms)``; zero-norm rows stay zero instead of producing NaNs."""
    norms = rows.norm(dim=-1, keepdim=True)
    safe = norms.clamp_min(eps)
    return rows / safe, norms.squeeze(-1)


_VERSION_UNREADABLE = -1  # sentinel: an inference tensor's _version cannot be read


@dataclass
class TurbovecIndexState:
    """CPU-resident turbovec index plus the bookkeeping needed to keep it in sync with a coreset.

    Not part of the node's persisted state: rebuilt from the coreset (deterministically, see the
    README) whenever ``coreset_signature`` no longer matches, so this struct never needs saving.
    """

    index: Any
    norms: Tensor  # [N] exact (unpadded) coreset row norms, CPU float32
    signature: tuple[int, torch.dtype, int, int, int, int]  # see _coreset_signature

    def matches(self, coreset: Tensor, generation: int) -> bool:
        return self.signature == _coreset_signature(coreset, generation)


def _coreset_signature(
    coreset: Tensor, generation: int
) -> tuple[int, torch.dtype, int, int, int, int]:
    """Cache key for a coreset buffer: shape, dtype, the node's own replacement generation, a
    version read (best-effort) and the buffer's storage address.

    ``generation`` is bumped by the node itself whenever it replaces or refits the coreset
    (including an ``assign=True`` state-dict load, which swaps in a new tensor object), so two
    different buffers of the same shape never share a signature even if torch's own version
    counter happens to coincide (two freshly-fitted tensors each mutated once both read version
    1). The buffer's ``data_ptr`` is included too, since a future caller that mutates ``coreset``
    directly (bypassing the node's own bookkeeping) is still caught by the storage address or the
    version even without a generation bump. ``_version`` is torch's private autograd counter and
    is unreadable on an inference tensor (``torch.inference_mode()``); that case reads as a
    constant sentinel instead of raising, which forces a rebuild on every call while such a
    tensor is in use rather than crashing or trusting a stale index.
    """
    try:
        version = coreset._version
    except RuntimeError:
        version = _VERSION_UNREADABLE
    return (
        coreset.shape[0],
        coreset.dtype,
        coreset.shape[1],
        generation,
        version,
        coreset.data_ptr(),
    )


def _import_turbovec() -> Any:
    try:
        import turbovec
    except ImportError as err:
        raise ImportError(
            "search_backend='turbovec' requires the 'turbovec' optional dependency. Install it "
            "with `uv pip install cuvis-ai-patchcore[turbovec]` (or `pip install "
            "cuvis-ai-patchcore[turbovec]`)."
        ) from err
    return turbovec


def validate_search_hparams(
    search_backend: str, search_bit_width: int, shortlist_size: int, rescore: bool
) -> None:
    if search_backend not in ("exact", "turbovec"):
        raise ValueError(f"search_backend must be 'exact' or 'turbovec', got {search_backend!r}")
    if search_bit_width not in _BIT_WIDTHS:
        raise ValueError(f"search_bit_width must be one of {_BIT_WIDTHS}, got {search_bit_width}")
    if search_backend == "exact" and not rescore:
        raise ValueError("rescore=False is not supported with search_backend='exact'")
    if shortlist_size < 1:
        raise ValueError(f"shortlist_size must be >= 1, got {shortlist_size}")


def build_turbovec_index(
    coreset: Tensor, bit_width: int, eps: float, generation: int
) -> TurbovecIndexState:
    """Build a CPU turbovec index from the current coreset rows.

    Rows are moved to the CPU, zero-padded to a multiple of 8 channels, unit-normalised and added
    to a fresh index; their exact (unpadded) norms are cached alongside it for rescoring and for
    the no-rescore reconstruction formula. ``generation`` is the node's own coreset-replacement
    counter, folded into the cached signature; see ``_coreset_signature``.
    """
    turbovec = _import_turbovec()
    cpu_coreset = coreset.detach().to("cpu", dtype=torch.float32)
    unit, norms = _unit_normalize(cpu_coreset, eps)
    padded = _pad_to_multiple_of_8(unit).contiguous().numpy()
    index = turbovec.TurboQuantIndex(dim=padded.shape[1], bit_width=bit_width)
    index.add(padded)
    return TurbovecIndexState(
        index=index, norms=norms, signature=_coreset_signature(coreset, generation)
    )


def turbovec_shortlist(
    state: TurbovecIndexState, flat: Tensor, shortlist_size: int, eps: float
) -> tuple[Tensor, Tensor]:
    """Top ``shortlist_size`` coreset row ids per query row by approximate cosine similarity.

    Returns ``(ids, scores)``, both ``[N, k]`` on the CPU (``k = min(shortlist_size,
    n_coreset_rows)``): ``ids`` is int64 coreset row indices, for the caller to gather fp32 rows
    back on the query device when rescoring exactly; ``scores`` is the index's own float32
    similarity for each returned row, the approximate cosine between the query and that coreset
    row (both sides are unit-normalised before being added to or searched against the index), to
    reuse directly when the caller does not rescore.
    """
    unit, _ = _unit_normalize(flat.detach().to("cpu", dtype=torch.float32), eps)
    padded = _pad_to_multiple_of_8(unit).contiguous().numpy()
    k = min(shortlist_size, len(state.index))
    scores, ids = state.index.search(padded, k)
    return torch.from_numpy(ids).to(torch.int64), torch.from_numpy(scores).to(torch.float32)


def reconstruct_cosine_distance(
    query_norms: Tensor, candidate_norms: Tensor, cos_sim: Tensor
) -> tuple[Tensor, int]:
    """Euclidean distance reconstructed from norms and an approximate cosine similarity.

    ``d = sqrt(max(0, |q|^2 + |c|^2 - 2|q||c|cos))``, with ``cos`` clamped to ``[-1, 1]`` first
    (the approximate search can return a value slightly outside that range). Returns
    ``(distance, n_clamped_negative)`` where the second element counts queries for which the
    radicand was negative before the ``max(0, ...)`` floor (diagnostics only).
    """
    cos = cos_sim.clamp(-1.0, 1.0)
    radicand = query_norms**2 + candidate_norms**2 - 2.0 * query_norms * candidate_norms * cos
    n_clamped = int((radicand < 0).sum().item())
    return radicand.clamp_min(0.0).sqrt(), n_clamped
