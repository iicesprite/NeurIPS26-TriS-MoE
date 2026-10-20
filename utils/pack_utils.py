import math

import torch


def pack_bool_vector(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    values = values.to(device=device, dtype=torch.bool).flatten()
    if values.numel() == 0:
        return torch.zeros((0,), dtype=torch.int32, device=device)
    pad = (-values.numel()) % 32
    if pad:
        values = torch.nn.functional.pad(values, (0, pad), value=False)
    shifts = torch.arange(32, dtype=torch.int64, device=device)
    chunks = values.view(-1, 32).to(torch.int64)
    packed = torch.sum(chunks << shifts.unsqueeze(0), dim=1)
    return packed.to(torch.int32).contiguous()


def pack_bool_matrix(matrix: torch.Tensor, device: torch.device) -> torch.Tensor:
    matrix = matrix.to(device=device, dtype=torch.bool).t().contiguous()
    if matrix.numel() == 0:
        return torch.zeros((0, matrix.shape[1]), dtype=torch.int32, device=device)
    pad = (-matrix.shape[0]) % 32
    if pad:
        matrix = torch.nn.functional.pad(matrix, (0, 0, 0, pad), value=False)
    shifts = torch.arange(32, dtype=torch.int64, device=device).view(1, 32, 1)
    chunks = matrix.view(-1, 32, matrix.shape[1]).to(torch.int64)
    packed = torch.sum(chunks << shifts, dim=1)
    return packed.to(torch.int32).contiguous()


def pack_sparse_bool_matrix(matrix: torch.Tensor, active_cols: torch.Tensor, device: torch.device) -> torch.Tensor:
    active_cols = active_cols.to(device=device, dtype=torch.bool).contiguous()
    if active_cols.numel() == 0 or int(active_cols.sum().item()) == 0:
        return torch.zeros((0, matrix.shape[0]), dtype=torch.int32, device=device)
    return pack_bool_matrix(matrix[:, active_cols], device)


def build_sparse_bit_prefix(active_cols: torch.Tensor, device: torch.device) -> torch.Tensor:
    active_cols_cpu = active_cols.detach().to(device="cpu", dtype=torch.bool).flatten()
    packed_rows = math.ceil(active_cols_cpu.numel() / 32)
    prefix = torch.zeros((packed_rows + 1,), dtype=torch.int32, device=device)
    if packed_rows == 0:
        return prefix
    running = 0
    values = [0]
    for packed_row in range(packed_rows):
        start = packed_row * 32
        end = min(start + 32, active_cols_cpu.numel())
        running += int(active_cols_cpu[start:end].sum().item())
        values.append(running)
    return torch.tensor(values, dtype=torch.int32, device=device)


def pack_half2_words(first: torch.Tensor, second: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    first_bits = first.contiguous().to(dtype=torch.float16).view(torch.int16).to(dtype=dtype) & 0xFFFF
    second_bits = second.contiguous().to(dtype=torch.float16).view(torch.int16).to(dtype=dtype) & 0xFFFF
    return first_bits | (second_bits << 16)


def pack_half4_words(first: torch.Tensor, second: torch.Tensor, third: torch.Tensor, fourth: torch.Tensor) -> torch.Tensor:
    low = pack_half2_words(first, second, torch.int64)
    high = pack_half2_words(third, fourth, torch.int64)
    return (low | (high << 32)).contiguous()


def build_compact3_packed_payload(
    *,
    device: torch.device,
    mode,
    b1,
    b2,
    g_map_non_salient,
    c_map,
    g_idx,
    scales_s_r1,
    scales_s_r2,
    scales_ns_d_r1,
    scales_ns_nd_r1,
    scales_s_c1,
    scales_s_c2,
    scales_ns_d_c1,
    scales_ns_nd_c1,
    ns_vector_row=None,
    ns_vector_col=None,
):
    folded_s_r1 = scales_s_r1
    folded_s_r2 = scales_s_r2
    folded_ns_d_r1 = scales_ns_d_r1
    folded_ns_nd_r1 = scales_ns_nd_r1
    folded_s_c1 = scales_s_c1
    folded_s_c2 = scales_s_c2
    folded_ns_d_c1 = scales_ns_d_c1
    folded_ns_nd_c1 = scales_ns_nd_c1

    if ns_vector_row is not None and ns_vector_col is not None:
        ns_row = ns_vector_row.to(device=device, dtype=torch.float16).unsqueeze(1)
        ns_col = ns_vector_col.to(device=device, dtype=torch.float16)
        folded_s_r1 = scales_s_r1 * ns_row
        folded_s_r2 = scales_s_r2 * ns_row
        folded_ns_d_r1 = scales_ns_d_r1 * ns_row
        folded_ns_nd_r1 = scales_ns_nd_r1 * ns_row
        folded_s_c1 = scales_s_c1 * ns_col
        folded_s_c2 = scales_s_c2 * ns_col
        folded_ns_d_c1 = scales_ns_d_c1 * ns_col
        folded_ns_nd_c1 = scales_ns_nd_c1 * ns_col

    non_salient = ~c_map.to(device=device, dtype=torch.bool)
    packed = {
        "mode": mode,
        "qweight1": pack_bool_matrix(b1, device),
        "qweight2_sparse": pack_sparse_bool_matrix(b2, c_map, device),
        "qweight2_sparse_prefix": build_sparse_bit_prefix(c_map, device),
        "q_gmap_sparse": pack_sparse_bool_matrix(g_map_non_salient, non_salient, device),
        "q_gmap_sparse_prefix": build_sparse_bit_prefix(non_salient, device),
        "c_map_packed": pack_bool_vector(c_map, device),
        "g_idx": g_idx.to(device=device, dtype=torch.int32).contiguous(),
        "row_pack3": pack_half4_words(folded_s_r1, folded_s_r2, folded_ns_d_r1, folded_ns_nd_r1),
        "col_pack3": pack_half4_words(folded_s_c1, folded_s_c2, folded_ns_d_c1, folded_ns_nd_c1),
    }
    if ns_vector_row is not None and ns_vector_col is not None:
        packed["ns_mode"] = "none"
        packed["ns_vector_row"] = None
        packed["ns_vector_col"] = None
    return packed


def build_packed_4bit_payload(matrix: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, g_idx: torch.Tensor) -> dict[str, torch.Tensor]:
    matrix_cpu = matrix.detach().float().cpu().contiguous()
    scale_cpu = scale.detach().float().cpu().contiguous()
    zero_cpu = zero.detach().float().cpu().contiguous()
    g_idx_cpu = g_idx.detach().to(dtype=torch.int32).cpu().contiguous()

    rows, columns = matrix_cpu.shape
    if scale_cpu.dim() != 2 or zero_cpu.dim() != 2:
        raise ValueError("Packed 4bit payload expects 2D scale/zero tensors.")

    if scale_cpu.shape[0] == rows:
        scales_group_major = scale_cpu.t().contiguous()
        zeros_group_major = zero_cpu.t().contiguous()
    elif scale_cpu.shape[1] == rows:
        scales_group_major = scale_cpu.contiguous()
        zeros_group_major = zero_cpu.contiguous()
    else:
        raise ValueError(
            f"Scale/zero shape mismatch for packed payload: matrix={matrix_cpu.shape}, "
            f"scale={scale_cpu.shape}, zero={zero_cpu.shape}"
        )

    if g_idx_cpu.numel() != columns:
        raise ValueError(f"g_idx length mismatch: expected {columns}, got {g_idx_cpu.numel()}")

    scales_for_cols = scales_group_major[g_idx_cpu.long()]
    zeros_for_cols = zeros_group_major[g_idx_cpu.long()]
    q_matrix = torch.round(matrix_cpu.t() / scales_for_cols + zeros_for_cols).clamp_(0, 15).to(torch.int64)

    qweight = torch.zeros((math.ceil(columns / 8), rows), dtype=torch.int32)
    for pack_col in range(qweight.shape[0]):
        start = pack_col * 8
        end = min(start + 8, columns)
        chunk = q_matrix[start:end]
        shifts = (torch.arange(end - start, dtype=torch.int64).view(-1, 1) * 4)
        qweight[pack_col] = torch.sum(chunk << shifts, dim=0).to(torch.int32)

    zeros_int = torch.round(zeros_group_major).clamp_(0, 15).to(torch.int64)
    zeros_int = torch.bitwise_and(zeros_int - 1, 0xF)
    qzeros = torch.zeros((zeros_int.shape[0], math.ceil(rows / 8)), dtype=torch.int32)
    for pack_row in range(qzeros.shape[1]):
        start = pack_row * 8
        end = min(start + 8, rows)
        chunk = zeros_int[:, start:end]
        shifts = (torch.arange(end - start, dtype=torch.int64).view(1, -1) * 4)
        qzeros[:, pack_row] = torch.sum(chunk << shifts, dim=1).to(torch.int32)

    return {
        "rows": torch.tensor(rows, dtype=torch.int32),
        "cols": torch.tensor(columns, dtype=torch.int32),
        "qweight": qweight.contiguous(),
        "qzeros": qzeros.contiguous(),
        "scales": scales_group_major.contiguous(),
        "g_idx": g_idx_cpu,
    }


def append_zero_column_to_packed_4bit_payload(payload: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    rows = int(payload["rows"].item())
    cols = int(payload["cols"].item())
    if cols % 2 == 0:
        return payload
    qweight = payload["qweight"].clone().to(torch.int32)
    qzeros = payload["qzeros"].clone().to(torch.int32)
    scales = payload["scales"].clone()
    g_idx = payload["g_idx"].clone().to(torch.int32)

    new_cols = cols + 1
    new_qweight = torch.zeros((math.ceil(new_cols / 8), rows), dtype=torch.int32)
    new_qweight[: qweight.shape[0], : qweight.shape[1]] = qweight
    pack_col = cols // 8
    shift = (cols & 7) * 4
    new_qweight[pack_col] |= (torch.ones((rows,), dtype=torch.int32) << shift)

    new_g_idx = torch.empty((new_cols,), dtype=torch.int32)
    new_g_idx[:cols] = g_idx
    new_g_idx[cols] = g_idx[-1] if g_idx.numel() > 0 else 0
    return {
        "rows": torch.tensor(rows, dtype=torch.int32),
        "cols": torch.tensor(new_cols, dtype=torch.int32),
        "qweight": new_qweight.contiguous(),
        "qzeros": qzeros.contiguous(),
        "scales": scales.contiguous(),
        "g_idx": new_g_idx.contiguous(),
    }


def append_exact_zero_row_to_packed_4bit_payload(payload: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    rows = int(payload["rows"].item())
    cols = int(payload["cols"].item())
    qweight = payload["qweight"].clone().to(torch.int32)
    qzeros = payload["qzeros"].clone().to(torch.int32)
    scales = payload["scales"].clone()
    g_idx = payload["g_idx"].clone().to(torch.int32)

    new_rows = rows + 1
    new_qweight = torch.zeros((qweight.shape[0], new_rows), dtype=torch.int32)
    new_qweight[:, :rows] = qweight
    for pack_col in range(qweight.shape[0]):
        start = pack_col * 8
        valid = max(0, min(8, cols - start))
        if valid == 0:
            continue
        packed_val = 0
        for lane in range(valid):
            packed_val |= (1 << (lane * 4))
        new_qweight[pack_col, rows] = packed_val

    new_qzeros = torch.zeros((qzeros.shape[0], math.ceil(new_rows / 8)), dtype=torch.int32)
    new_qzeros[:, : qzeros.shape[1]] = qzeros

    new_scales = torch.ones((scales.shape[0], new_rows), dtype=scales.dtype)
    new_scales[:, :rows] = scales

    return {
        "rows": torch.tensor(new_rows, dtype=torch.int32),
        "cols": torch.tensor(cols, dtype=torch.int32),
        "qweight": new_qweight.contiguous(),
        "qzeros": new_qzeros.contiguous(),
        "scales": new_scales.contiguous(),
        "g_idx": g_idx.contiguous(),
    }


def pad_nullspace_payloads_even_rank(
    c1_payload: dict[str, torch.Tensor],
    c2_payload: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if int(c1_payload["cols"].item()) % 2 == 0:
        return c1_payload, c2_payload
    return (
        append_zero_column_to_packed_4bit_payload(c1_payload),
        append_exact_zero_row_to_packed_4bit_payload(c2_payload),
    )
