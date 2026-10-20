"""Spectral bases, row/column scaling, and packed low-rank weight corrections."""

import logging

import torch
import torch.nn as nn
import transformers

from utils.pack_utils import (
    append_exact_zero_row_to_packed_4bit_payload,
    append_zero_column_to_packed_4bit_payload,
    build_packed_4bit_payload,
    pad_nullspace_payloads_even_rank,
)

logger = logging.getLogger()


class SpectralCorrectionMixin:
    def _should_force_cpu_eigh(self):
        return (
            self.build_deploy_payload
            and self.is_expert_layer
            and self.model_type.startswith("qwen3_5_moe")
        )

    def _run_eigh(self, X, context):
        eig_input = X
        if self._should_force_cpu_eigh() and X.device.type != "cpu":
            logger.info(f"Running CPU eigh fallback for {context}")
            eig_input = X.cpu()

        eigvals, eigvecs = torch.linalg.eigh(eig_input)

        if eig_input.device != X.device:
            eigvals = eigvals.to(device=X.device)
            eigvecs = eigvecs.to(device=X.device)

        return eigvals, eigvecs

    def _prepare_symmetric_second_moment(self, avg_x, context):
        if avg_x is None:
            return None

        X = 0.5 * (avg_x.float() + avg_x.float().t())

        if X.ndim != 2 or X.shape[0] != X.shape[1] or X.numel() == 0:
            logger.warning(
                f"Skipping null-space eig for {context}: invalid matrix shape={tuple(X.shape)}"
            )
            return None

        finite_mask = torch.isfinite(X)
        if not bool(finite_mask.all().item()):
            nan_count = int(torch.isnan(X).sum().item())
            inf_count = int(torch.isinf(X).sum().item())
            finite_count = int(finite_mask.sum().item())
            finite_min = None
            finite_max = None
            if finite_count > 0:
                finite_values = X[finite_mask]
                finite_min = float(finite_values.min().item())
                finite_max = float(finite_values.max().item())
            logger.warning(
                f"Skipping null-space eig for {context}: non-finite input_second_moment matrix "
                f"shape={tuple(X.shape)}, nan_count={nan_count}, inf_count={inf_count}, "
                f"finite_min={finite_min}, finite_max={finite_max}"
            )
            return None

        return (0.5 * (X + X.t())).contiguous()

    def _prepare_shared_correction_basis(self):
        if not self._uses_shared_gate_up_basis():
            return None

        data = self.factorization_cache.get(self.projection_name)
        if data is None:
            return None
        if data.get("ns_u2_global") is not None:
            return data

        avg_x = self.shared_input_second_moment
        if avg_x is None:
            logger.info(f"Shared gate/up null-space basis unavailable for {self.layer_name}; falling back to local null-space basis.")
            return None
        X = self._build_null_space_eigh_matrix(avg_x, f"{self.layer_name} shared gate/up")
        if X is None:
            return None
        eigvals, eigvecs = self._run_eigh(X, f"{self.layer_name} shared gate/up")
        sorted_indices = torch.argsort(eigvals, descending=True)
        s = eigvals[sorted_indices]
        u = eigvecs[:, sorted_indices]
        index = self._select_ns_index(s)
        logger.info(f"shared ns_index in {self.layer_name}: {index}")
        u2_global = u[:, : index + 1].contiguous()
        c2 = u2_global.t().contiguous()

        H_shared = self.hessian_for_shared_ns
        if H_shared is None and hasattr(self, 'hessian_for_shared_svd') and self.hessian_for_shared_svd is not None:
            H_shared = self.hessian_for_shared_svd
        if H_shared is None:
            H_shared = self.H
        H_shared = H_shared.clone().float()

        c2_quant, c2_scale, c2_zero, c2_g_idx = self._gptq_quantize_ns(c2, H_shared)
        c2_payload = None
        if self.build_deploy_payload:
            c2_payload = build_packed_4bit_payload(c2_quant, c2_scale, c2_zero, c2_g_idx)
            if int(c2_payload["rows"].item()) % 2 != 0:
                c2_payload = append_exact_zero_row_to_packed_4bit_payload(c2_payload)

        data["ns_u2_global"] = u2_global
        data["ns_h1_shared"] = u2_global.t() @ H_shared @ u2_global
        data["ns_c2_shared"] = c2_quant
        data["ns_c2_shared_quant"] = c2_payload
        data["ns_rank_shared"] = u2_global.shape[1]
        return data

    def _apply_shared_basis_correction(self, W_reconstructed):
        if not self._correction_applies(self.layer_name or ''):
            return W_reconstructed, None

        data = self._build_shared_gate_up_ns_basis()
        if data is None or data.get("ns_u2_global") is None or data.get("ns_c2_shared") is None:
            return self._apply_null_space_optimization_with_matrix(W_reconstructed)

        W_original = self.layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d):
            W_original = W_original.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W_original = W_original.t()

        u2_global = data["ns_u2_global"]
        c2_shared = data["ns_c2_shared"]
        c1 = (W_original - W_reconstructed) @ u2_global
        c1, c1_scale, c1_zero, c1_g_idx = self._gptq_quantize_ns(
            c1,
            data["ns_h1_shared"],
            groupsize=-1,
        )

        W_compensation = c1 @ c2_shared
        c_weight = None
        if self.build_deploy_payload:
            c1_payload = build_packed_4bit_payload(c1, c1_scale, c1_zero, c1_g_idx)
            shared_c2_payload = data["ns_c2_shared_quant"]
            if shared_c2_payload is not None and int(c1_payload["cols"].item()) != int(shared_c2_payload["rows"].item()):
                c1_payload = append_zero_column_to_packed_4bit_payload(c1_payload)
            c_weight = {
                "c1": c1_payload,
                "shared_c2": True,
            }
        W_optimized = W_reconstructed + W_compensation
        return W_optimized, c_weight

    def _fit_row_column_correction(self, W_reconstructed):
        if not self._correction_applies(self.layer_name or ''):
            return W_reconstructed, None, None
        if self.input_second_moment is None:
            logger.info(f"input_second_moment is none for {self.layer_name}, skip vector ns.")
            return W_reconstructed, None, None

        lambda_t = 0.2

        X = self._build_null_space_eigh_matrix(self.input_second_moment, f"{self.layer_name} vector")
        if X is None:
            return W_reconstructed, None, None

        eigvals, eigvecs = self._run_eigh(X, f"{self.layer_name} vector")
        sorted_indices = torch.argsort(eigvals, descending=True)
        s = eigvals[sorted_indices]
        u = eigvecs[:, sorted_indices]

        index = self._select_ns_index(s)

        u_zero = u[:, index:]
        p = u_zero @ u_zero.T

        W_original = self.layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d):
            W_original = W_original.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W_original = W_original.t()

        with torch.enable_grad():
            I = torch.eye(W_reconstructed.shape[1], device=self.dev)
            B = W_original @ (I - p) + W_reconstructed @ p

            beta1 = torch.ones(W_reconstructed.shape[0], 1, device=self.dev, dtype=W_reconstructed.dtype)
            beta2 = torch.ones(1, W_reconstructed.shape[1], device=self.dev, dtype=W_reconstructed.dtype)

            max_iter = 50
            tol = 1e-5
            for iter_idx in range(max_iter):
                Wq_beta2 = W_reconstructed * beta2
                numerator_beta1 = torch.sum(B * Wq_beta2, dim=1) + lambda_t
                denominator_beta1 = torch.sum(Wq_beta2**2, dim=1) + lambda_t
                beta1_new = numerator_beta1 / (denominator_beta1 + 1e-8)
                beta1_new = beta1_new.unsqueeze(1)

                Wq_beta1 = beta1_new * W_reconstructed
                numerator_beta2 = torch.sum(B * Wq_beta1, dim=0) + lambda_t
                denominator_beta2 = torch.sum(Wq_beta1**2, dim=0) + lambda_t
                beta2_new = numerator_beta2 / (denominator_beta2 + 1e-8)
                beta2_new = beta2_new.unsqueeze(0)

                if torch.norm(beta1_new - beta1) < tol and torch.norm(beta2_new - beta2) < tol:
                    break
                beta1, beta2 = beta1_new, beta2_new

            W_optimized = beta1 * W_reconstructed * beta2
        return W_optimized, beta1.squeeze(1), beta2.squeeze(0)

    def _apply_low_rank_correction(self, W_reconstructed):
        if not self._correction_applies(self.layer_name or ''):
            return W_reconstructed, None

        if self.input_second_moment == None:
            logger.info(f"input_second_moment is none for {self.layer_name}, skip matrix ns.")
            return W_reconstructed, None

        X = self._build_null_space_eigh_matrix(self.input_second_moment, f"{self.layer_name} matrix")
        if X is None:
            return W_reconstructed, None

        eigvals, eigvecs = self._run_eigh(X, f"{self.layer_name} matrix")
        sorted_indices = torch.argsort(eigvals, descending=True)
        s = eigvals[sorted_indices]
        u = eigvecs[:, sorted_indices]

        index = self._select_ns_index(s)

        W_original = self.layer.weight.data.clone().float()

        u_master = u[:, :index+1]
        c1 = (W_original - W_reconstructed) @ u_master
        c2 = u_master.T

        c2, c2_scale, c2_zero, c2_g_idx = self._gptq_quantize_ns(c2, self.H)
        H_orig = self.H
        H1 = u_master.T @ H_orig @ u_master
        c1, c1_scale, c1_zero, c1_g_idx = self._gptq_quantize_ns(c1, H1)

        if isinstance(self.layer, nn.Conv2d):
            W_original = W_original.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W_original = W_original.t()

        W_compensation=c1 @ c2
        c1_payload = build_packed_4bit_payload(c1, c1_scale, c1_zero, c1_g_idx)
        c2_payload = build_packed_4bit_payload(c2, c2_scale, c2_zero, c2_g_idx)
        c1_payload, c2_payload = pad_nullspace_payloads_even_rank(c1_payload, c2_payload)
        c_weight = [c1_payload, c2_payload]
        W_optimized = W_reconstructed + W_compensation
        return W_optimized, c_weight
