"""Compatibility imports for the established spectrum-selection API."""
from tris_moe.quantization.rank_selection import (
    torch,
    np,
    curve_fit,
    select_rank_by_tail_energy,
    find_max_curvature_index,
    find_max_kneedle_index,
    model,
    find_max_fit_index,
)


def fast_find_energy_drop_index(S, threshold=0.05):
    return select_rank_by_tail_energy(S, threshold=threshold)


def find_max_fit_nohead_index(s: torch.Tensor, layer_name) -> int:
    # Keep lookup of find_max_fit_index in this module for existing callers that
    # replace the fitting policy through the compatibility module.
    if s.numel() <= 1:
        return 0
    if s.numel() <= 3:
        return 1
    index = find_max_fit_index(s[1:], layer_name)
    if index is None:
        return 0
    return int(index) + 1
