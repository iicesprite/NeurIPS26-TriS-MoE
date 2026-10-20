"""ARB-RC block quantization for dense weights and expert left factors."""

import logging

import torch
import torch.nn as nn
import transformers

from utils.structure import structural_guassian_distribution_multip_alternating_group_x
from utils.pack_utils import build_compact3_packed_payload

logger = logging.getLogger()


class BlockwiseQuantizationMixin:
    def _quantize_expert_left_factor(self, expert_idx, blocksize, percdamp, orders, num_p):
        data = self.factorization_cache.get(self.projection_name)

        expert_left_factors = data['expert_Us']
        if expert_idx >= len(expert_left_factors):
            logger.error(f"Expert index {expert_idx} out of range")
            return None, 0

        left_factor = expert_left_factors[expert_idx].clone()
        shared_right_factor = data['SV_shared'].float()
        input_hessian = self.H
        factor_hessian = shared_right_factor @ input_hessian @ shared_right_factor.t()

        rows, columns = left_factor.shape
        compact3_mode = num_p == 1
        can_pack_payload = self.build_deploy_payload and compact3_mode

        dead = torch.diag(factor_hessian) == 0
        factor_hessian[dead, dead] = 1
        left_factor[:, dead] = 0
        damp = percdamp * torch.mean(torch.diag(factor_hessian))
        diag = torch.arange(columns, device=self.dev)
        factor_hessian[diag, diag] += damp
        factor_hessian = torch.linalg.cholesky(factor_hessian)
        factor_hessian = torch.cholesky_inverse(factor_hessian)
        factor_hessian = torch.linalg.cholesky(factor_hessian, upper=True)
        factor_hessian_inv = factor_hessian

        Losses = torch.zeros(rows, device=self.dev)
        quantized_left_factor = torch.zeros_like(left_factor)
        g_idx = torch.tensor([i // blocksize for i in range(columns)], dtype=torch.int32, device=self.dev)
        B1 = None
        B2 = None
        g_map = None
        c_map = None
        compact3_s_c1 = None
        compact3_s_r1 = None
        compact3_s_c2 = None
        compact3_s_r2 = None
        compact3_ns_d_c1 = None
        compact3_ns_d_r1 = None
        compact3_ns_nd_c1 = None
        compact3_ns_nd_r1 = None
        if can_pack_payload:
            n_groups = int(g_idx[-1].item()) + 1
            B1 = torch.zeros_like(left_factor)
            B2 = torch.zeros_like(left_factor)
            g_map = torch.zeros_like(left_factor, dtype=torch.bool)
            c_map = torch.zeros((columns,), device=self.dev, dtype=torch.bool)
            compact3_s_c1 = torch.zeros((columns,), device=self.dev)
            compact3_s_r1 = torch.zeros((rows, n_groups), device=self.dev)
            compact3_s_c2 = torch.zeros((columns,), device=self.dev)
            compact3_s_r2 = torch.zeros((rows, n_groups), device=self.dev)
            compact3_ns_d_c1 = torch.zeros((columns,), device=self.dev)
            compact3_ns_d_r1 = torch.zeros((rows, n_groups), device=self.dev)
            compact3_ns_nd_c1 = torch.zeros((columns,), device=self.dev)
            compact3_ns_nd_r1 = torch.zeros((rows, n_groups), device=self.dev)

        for col_st in range(0, columns, blocksize):
            col_ed = min(col_st + blocksize, columns)

            num_mask = 1 + num_p + 1
            orders_list = [2] + [1] * (num_p + 1)

            mask = torch.zeros_like(left_factor[:, col_st:col_ed], dtype=torch.bool).unsqueeze(0).repeat_interleave(num_mask, dim=0)

            block_metric = torch.eye(col_ed - col_st, device=self.dev, dtype=torch.float32)
            mask_list = structural_guassian_distribution_multip_alternating_group_x(
                left_factor[:, col_st:col_ed], factor_hessian[col_st:col_ed, col_st:col_ed],
                64, num_p, block_metric
            )
            for i in range(num_mask):
                mask[i] = mask_list[i]

            if self.disable_gptq:
                block_weights = left_factor[:, col_st:col_ed]
                q_part_groups = [self.braq_quantizer.quantize(column, mask[i], order=orders_list[i]) for i in range(mask.shape[0])]
                q = torch.zeros_like(block_weights)
                for j in range(mask.shape[0]):
                    q += q_part_groups[j][:] * mask[j, :]
                left_factor[:, col_st:col_ed] = q
                quantized_left_factor[:, col_st:col_ed] = q
            else:
                factor_block = left_factor[:, col_st:col_ed].clone()
                quantized_block = torch.zeros_like(factor_block)
                B1_block = torch.zeros_like(factor_block) if can_pack_payload else None
                B2_block = torch.zeros_like(factor_block) if can_pack_payload else None
                block_error = torch.zeros_like(factor_block)
                block_losses = torch.zeros_like(factor_block)
                block_hessian_inv = factor_hessian_inv[col_st:col_ed, col_st:col_ed]

                q_part_groups = []
                block_binaries = []
                block_alpha_r = []
                block_alpha_c = []
                for idx in range(mask.shape[0]):
                    u_approx, alpha_list_c, alpha_list_r, binary_list = self.braq_quantizer.quantize(
                        factor_block, mask[idx], order=orders_list[idx], S=block_metric
                    )
                    q_part_groups.append(u_approx)
                    block_binaries.append(binary_list)
                    block_alpha_r.append(alpha_list_r)
                    block_alpha_c.append(alpha_list_c)

                if can_pack_payload:
                    block_group = int(g_idx[col_st].item())
                    compact3_s_c1[col_st:col_ed] = block_alpha_c[0][0].reshape(-1)
                    compact3_s_r1[:, block_group] = block_alpha_r[0][0].reshape(-1)
                    compact3_s_c2[col_st:col_ed] = block_alpha_c[0][1].reshape(-1)
                    compact3_s_r2[:, block_group] = block_alpha_r[0][1].reshape(-1)
                    compact3_ns_d_c1[col_st:col_ed] = block_alpha_c[1][0].reshape(-1)
                    compact3_ns_d_r1[:, block_group] = block_alpha_r[1][0].reshape(-1)
                    compact3_ns_nd_c1[col_st:col_ed] = block_alpha_c[2][0].reshape(-1)
                    compact3_ns_nd_r1[:, block_group] = block_alpha_r[2][0].reshape(-1)

                for i in range(col_ed - col_st):
                    column = factor_block[:, i]
                    d = block_hessian_inv[i, i]
                    q = torch.zeros_like(column)
                    for j in range(mask.shape[0]):
                        q += q_part_groups[j][:, i] * mask[j, :, i]
                    quantized_block[:, i] = q
                    if can_pack_payload:
                        b1 = torch.zeros_like(column)
                        for j in range(mask.shape[0]):
                            b1 += block_binaries[j][0][:, i] * mask[j, :, i]
                        b2 = block_binaries[0][1][:, i] * mask[0, :, i]
                        B1_block[:, i] = b1
                        B2_block[:, i] = b2
                    block_losses[:, i] = (column - q) ** 2 / d ** 2
                    err1 = (column - q) / d
                    factor_block[:, i:] -= err1.unsqueeze(1).matmul(block_hessian_inv[i, i:].unsqueeze(0))
                    block_error[:, i] = err1

                quantized_left_factor[:, col_st:col_ed] = quantized_block
                if can_pack_payload:
                    B1[:, col_st:col_ed] = B1_block
                    B2[:, col_st:col_ed] = B2_block
                    c_map[col_st:col_ed] = mask_list[0].any(dim=0)
                    g_map[:, col_st:col_ed] = mask_list[1]
                Losses += torch.sum(block_losses, 1) / 2
                left_factor[:, col_ed:] -= block_error.matmul(factor_hessian_inv[col_st:col_ed, col_ed:])

        if 'expert_Us_quantized' not in data:
            data['expert_Us_quantized'] = [None] * data['num_experts']
        data['expert_Us_quantized'][expert_idx] = quantized_left_factor

        quant_error = torch.sum(Losses).item()
        payload = None
        if compact3_mode and can_pack_payload:
            payload = build_compact3_packed_payload(
                device=self.dev,
                mode='expert_compact3',
                b1=(B1 > 0),
                b2=(B2 > 0),
                g_map_non_salient=g_map,
                c_map=c_map,
                g_idx=g_idx,
                scales_s_r1=compact3_s_r1,
                scales_s_r2=compact3_s_r2,
                scales_ns_d_r1=compact3_ns_d_r1,
                scales_ns_nd_r1=compact3_ns_nd_r1,
                scales_s_c1=compact3_s_c1,
                scales_s_c2=compact3_s_c2,
                scales_ns_d_c1=compact3_ns_d_c1,
                scales_ns_nd_c1=compact3_ns_nd_c1,
            )
        return quantized_left_factor, quant_error, payload

    def _quantize_dense_weight(self, blocksize, percdamp, orders, num_p):
        """Non-expert layer quantization"""
        results = {}
        compact3_mode = num_p == 1
        can_pack_payload = self.build_deploy_payload and compact3_mode

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        try:
            H = self.H.clone()
            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            W[:, dead] = 0

            Losses = torch.zeros(self.rows, device=self.dev)
            Q = torch.zeros_like(W)
            B1 = None
            B2 = None
            g_map = None
            c_map = None

            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=self.dev)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H

            num_mask = 1 + num_p + 1
            orders_list = [2] + [1] * (num_p + 1)

            g_idx = torch.tensor([i // blocksize for i in range(self.columns)], dtype=torch.int32, device=self.dev)
            compact3_s_c1 = None
            compact3_s_r1 = None
            compact3_s_c2 = None
            compact3_s_r2 = None
            compact3_ns_d_c1 = None
            compact3_ns_d_r1 = None
            compact3_ns_nd_c1 = None
            compact3_ns_nd_r1 = None
            if can_pack_payload:
                n_groups = int(g_idx[-1].item()) + 1
                B1 = torch.zeros_like(W)
                B2 = torch.zeros_like(W)
                g_map = torch.zeros_like(W, dtype=torch.bool)
                c_map = torch.zeros((self.columns,), dtype=torch.bool, device=self.dev)
                compact3_s_c1 = torch.zeros((self.columns,), device=self.dev)
                compact3_s_r1 = torch.zeros((W.shape[0], n_groups), device=self.dev)
                compact3_s_c2 = torch.zeros((self.columns,), device=self.dev)
                compact3_s_r2 = torch.zeros((W.shape[0], n_groups), device=self.dev)
                compact3_ns_d_c1 = torch.zeros((self.columns,), device=self.dev)
                compact3_ns_d_r1 = torch.zeros((W.shape[0], n_groups), device=self.dev)
                compact3_ns_nd_c1 = torch.zeros((self.columns,), device=self.dev)
                compact3_ns_nd_r1 = torch.zeros((W.shape[0], n_groups), device=self.dev)

            for col_st in range(0, self.columns, blocksize):
                col_ed = min(col_st + blocksize, self.columns)

                S = None
                mask = torch.zeros_like(W[:, col_st:col_ed], dtype=torch.bool).unsqueeze(0).repeat_interleave(num_mask, dim=0)
                mask_list = structural_guassian_distribution_multip_alternating_group_x(
                    W[:, col_st:col_ed], H[col_st:col_ed, col_st:col_ed],
                    64, num_p, S
                )
                for i in range(num_mask):
                    mask[i] = mask_list[i]

                if self.disable_gptq:
                    w = W[:, col_st:col_ed]
                    q_part_groups = [self.braq_quantizer.quantize(w, mask[i], order=orders_list[i]) for i in range(mask.shape[0])]
                    q = torch.zeros_like(w)
                    for j in range(mask.shape[0]):
                        q += q_part_groups[j][:] * mask[j, :]
                    W[:, col_st:col_ed] = q
                else:
                    W1 = W[:, col_st:col_ed].clone()
                    Q1 = torch.zeros_like(W1)
                    B1_block = torch.zeros_like(W1) if can_pack_payload else None
                    B2_block = torch.zeros_like(W1) if can_pack_payload else None
                    Err1 = torch.zeros_like(W1)
                    Losses1 = torch.zeros_like(W1)
                    Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

                    q_part_groups = []
                    block_binaries = []
                    block_alpha_r = []
                    block_alpha_c = []

                    for idx in range(mask.shape[0]):
                        w_approx, alpha_list_c, alpha_list_r, binary_list = self.braq_quantizer.quantize(
                            W1, mask[idx], order=orders_list[idx], S=S
                        )
                        q_part_groups.append(w_approx)
                        block_binaries.append(binary_list)
                        block_alpha_r.append(alpha_list_r)
                        block_alpha_c.append(alpha_list_c)

                    if can_pack_payload:
                        block_group = int(g_idx[col_st].item())
                        compact3_s_c1[col_st:col_ed] = block_alpha_c[0][0].reshape(-1)
                        compact3_s_r1[:, block_group] = block_alpha_r[0][0].reshape(-1)
                        compact3_s_c2[col_st:col_ed] = block_alpha_c[0][1].reshape(-1)
                        compact3_s_r2[:, block_group] = block_alpha_r[0][1].reshape(-1)
                        compact3_ns_d_c1[col_st:col_ed] = block_alpha_c[1][0].reshape(-1)
                        compact3_ns_d_r1[:, block_group] = block_alpha_r[1][0].reshape(-1)
                        compact3_ns_nd_c1[col_st:col_ed] = block_alpha_c[2][0].reshape(-1)
                        compact3_ns_nd_r1[:, block_group] = block_alpha_r[2][0].reshape(-1)

                    for i in range(col_ed - col_st):
                        w = W1[:, i]
                        d = Hinv1[i, i]
                        q = torch.zeros_like(w)
                        for j in range(mask.shape[0]):
                            q += q_part_groups[j][:, i] * mask[j, :, i]

                        Q1[:, i] = q
                        if can_pack_payload:
                            b1 = torch.zeros_like(w)
                            for j in range(mask.shape[0]):
                                b1 += block_binaries[j][0][:, i] * mask[j, :, i]
                            b2 = block_binaries[0][1][:, i] * mask[0, :, i]
                            B1_block[:, i] = b1
                            B2_block[:, i] = b2
                        Losses1[:, i] = (w - q) ** 2 / d ** 2
                        err1 = (w - q) / d
                        W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                        Err1[:, i] = err1

                    Q[:, col_st:col_ed] = Q1
                    if can_pack_payload:
                        B1[:, col_st:col_ed] = B1_block
                        B2[:, col_st:col_ed] = B2_block
                        c_map[col_st:col_ed] = mask_list[0].any(dim=0)
                        g_map[:, col_st:col_ed] = mask_list[1]

                    Losses += torch.sum(Losses1, 1) / 2
                    W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])
            ns_vector_row = None
            ns_vector_col = None
            c_weight = None
            if self.ns_rule != "disable":
                try:
                    if isinstance(self.layer, transformers.Conv1D):
                        Q_for_null_space = Q.t()
                    else:
                        Q_for_null_space = Q

                    if self.ns_rule == "matrix":
                        Q_optimized, c_weight = self._apply_null_space_optimization_with_matrix(Q_for_null_space)
                    elif self.ns_rule == "vector":
                        Q_optimized, ns_vector_row, ns_vector_col = self._fit_row_column_correction(Q_for_null_space)
                    else:
                        Q_optimized = Q_for_null_space

                    if isinstance(self.layer, transformers.Conv1D):
                        Q = Q_optimized.t()
                    else:
                        Q = Q_optimized
                except Exception as e:
                    logger.warning(f"Null space optimization failed for layer {self.layer_name}: {e}")

            if isinstance(self.layer, transformers.Conv1D):
                Q = Q.t()
            self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
                self.layer.weight.data.dtype
            )

            if hasattr(self, 'inp'):
                del self.inp
            self._empty_device_cache(self.dev)

            results["error"] = torch.sum(Losses).item()
            if compact3_mode:
                results["mode"] = "compact3"
            else:
                results["mode"] = "dense"

            if compact3_mode and can_pack_payload:
                packed = build_compact3_packed_payload(
                    device=self.dev,
                    mode='compact3',
                    b1=(B1 > 0),
                    b2=(B2 > 0),
                    g_map_non_salient=g_map.bool(),
                    c_map=c_map.bool(),
                    g_idx=g_idx,
                    scales_s_r1=compact3_s_r1,
                    scales_s_r2=compact3_s_r2,
                    scales_ns_d_r1=compact3_ns_d_r1,
                    scales_ns_nd_r1=compact3_ns_nd_r1,
                    scales_s_c1=compact3_s_c1,
                    scales_s_c2=compact3_s_c2,
                    scales_ns_d_c1=compact3_ns_d_c1,
                    scales_ns_nd_c1=compact3_ns_nd_c1,
                    ns_vector_row=ns_vector_row,
                    ns_vector_col=ns_vector_col,
                )
                results.update(packed)
                results["ns_mode"] = "none"
                results["ns_vector_row"] = None
                results["ns_vector_col"] = None
                results["c_weight"] = None

            if self.ns_rule == "matrix" and c_weight is not None:
                results["ns_mode"] = "matrix"
                results["c_weight"] = c_weight
            elif (
                (not compact3_mode)
                and self.ns_rule == "vector"
                and ns_vector_row is not None
                and ns_vector_col is not None
            ):
                results["ns_mode"] = "vector"
                results["ns_vector_row"] = ns_vector_row
                results["ns_vector_col"] = ns_vector_col

            return results

        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            results["error"] = float('inf')
            return results
