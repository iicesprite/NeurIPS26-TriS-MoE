"""Spectrum cutoff policies; cutoff conventions match the public compatibility API."""

import torch
import numpy as np
from scipy.optimize import curve_fit

def select_rank_by_tail_energy(S, threshold=0.05):
    S = S.flatten()[1:]
    S = torch.clamp(S, min=0)
    prefix_sum = torch.cumsum(S, dim=0)
    total_sum = prefix_sum[-1]
    suffix_sum = total_sum - prefix_sum
    valid = prefix_sum[:-1] > 1e-12
    ratio = suffix_sum[1:] / prefix_sum[:-1]
    mask = (ratio <= threshold) & valid
    if torch.any(mask):
        i = torch.nonzero(mask)[0].item() + 1
        return i + 1
    else:
        return None

def find_max_curvature_index(s: torch.Tensor,layer_name) -> int:
    s = s.flatten()
    s = torch.clamp(s, min=0)
    n = len(s)

    if n < 3:
        return 0

    s_sq = s ** 2
    E = torch.cumsum(s_sq, dim=0) / torch.sum(s_sq)

    indices = torch.arange(n, dtype=torch.float32, device=s.device)
    x = indices / (n - 1)

    E_min = E[0]
    E_max = E[-1]

    if E_max == E_min:
        return 0

    y = (E - E_min) / (E_max - E_min)

    dx = 1.0 / (n - 1)

    # Use PyTorch finite differences for first and second derivatives.
    y_prime = torch.gradient(y, spacing=dx)[0]
    y_double_prime = torch.gradient(y_prime, spacing=dx)[0]

    # Curvature: kappa = |y''| / (1 + (y')^2)^(3/2).
    numerator = torch.abs(y_double_prime)
    denominator = torch.pow(1.0 + torch.pow(y_prime, 2), 1.5)
    kappa = numerator / denominator

    kappa[0] = 0
    kappa[-1] = 0

    max_curvature_idx = torch.argmax(kappa).item()

    #plot_smoothed_energy_values(E,f"energy_values_plot/{layer_name}.png",max_curvature_idx)
    return max_curvature_idx

def find_max_kneedle_index(s: torch.Tensor,layer_name) -> int:
    s = s.flatten()
    s = torch.clamp(s, min=0)
    n = len(s)

    if n < 3:
        return 0

    s_sq = s ** 2
    E = torch.cumsum(s_sq, dim=0) / torch.sum(s_sq)

    indices = torch.arange(n, dtype=torch.float32, device=s.device)
    x_norm = indices / n

    E_min = E[0]
    E_max = E[-1]

    if E_max == E_min:
        return 0

    y = (E - E_min) / (E_max - E_min)

    g = y - x_norm
    max_idx = torch.argmax(g)

    #plot_smoothed_energy_values(E,f"energy_values_plot_with_kneedle/{layer_name}.png",max_idx)
    return max_idx

def model(x, a, b):
    return 1 - a * np.exp(-b * x)

def find_max_fit_index(s: torch.Tensor,layer_name) -> int:
    s = s.flatten()
    s = torch.clamp(s, min=0)

    n = len(s)

    s_sq = s ** 2
    E = torch.cumsum(s_sq, dim=0) / torch.sum(s_sq)

    x = np.arange(n, dtype=np.float32)

    x_norm = x / x.max()

    E_min = E[0]
    E_max = E[-1]

    y = (E - E_min) / (E_max - E_min)
    y = y.detach().cpu().numpy()

    params, _ = curve_fit(
        model,
        x_norm,
        y,
        p0=[1.0, 10.0],
        maxfev=1000,
        #bounds=([0, 0], [2, 100])
    )

    a, b = params

    y_fit = model(x_norm, a, b)

    dy = np.gradient(y_fit, x_norm)

    d2y = np.gradient(dy, x_norm)

    kappa = np.abs(d2y) / (1 + dy**2)**1.5

    idx = np.argmax(kappa)
    idx = torch.tensor(idx, dtype=torch.int, device=s.device)
    #plot_smoothed_energy_values(E,f"energy_values_plot_with_fit/{layer_name}.png",idx)
    return idx

def find_max_fit_nohead_index(s: torch.Tensor, layer_name) -> int:
    if s.numel() <= 1:
        return 0
    if s.numel() <= 3:
        return 1
    index = find_max_fit_index(s[1:], layer_name)
    if index is None:
        return 0
    return int(index) + 1
