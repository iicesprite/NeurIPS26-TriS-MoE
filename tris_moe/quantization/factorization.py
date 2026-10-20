"""Joint factorization into expert left factors and a shared quantized right factor."""

import logging
from dataclasses import dataclass

import torch
import transformers

import torch.linalg as linalg
from utils.pack_utils import build_packed_4bit_payload

logger = logging.getLogger()


class ProjectionFactorStore:
    """A live view of one projection's factors in the public cache dictionary.

    The dictionary and its tensors are shared with existing quantizers and
    checkpoint code. This view never copies or replaces that dictionary.
    """

    def __init__(self, entries, projection):
        self.entries = entries
        self.projection = projection

    def ensure_ready(self, initialize):
        return self.projection in self.entries or initialize()

    def publish(self, factors):
        self.entries[self.projection] = factors

    def reconstruct(self, expert_index):
        factors = self.entries.get(self.projection)
        if not factors or 'expert_Us_quantized' not in factors:
            logger.error("Quantized expert left factors not found")
            return None
        right = factors['SV_shared']
        left = factors['expert_Us_quantized']
        if expert_index >= len(left) or left[expert_index] is None:
            logger.error(f"Quantized U matrix for expert {expert_index} not found")
            return None
        return left[expert_index] @ right

    def export_bank(self, to_cpu):
        factors = self.entries.get(self.projection)
        if factors is None or factors.get("SV_shared") is None:
            return None
        bank = {"shared_sv": to_cpu(factors.get("SV_shared_quant") or factors["SV_shared"])}
        if factors.get("ns_c2_shared_quant") is not None or factors.get("ns_c2_shared") is not None:
            bank["shared_ns_c2"] = to_cpu(factors.get("ns_c2_shared_quant") or factors["ns_c2_shared"])
        return bank

    def release(self):
        self.entries.clear()


@dataclass
class RowStackDecomposition:
    left_vectors: torch.Tensor
    right_factor: torch.Tensor
    rank: int
    expert_count: int
    output_width: int
    input_width: int
    dtype: torch.dtype
    device: torch.device

    @classmethod
    def compute(cls, weights):
        count = len(weights)
        output_width, input_width = weights[0].shape
        dtype, device = weights[0].dtype, weights[0].device
        stacked = torch.cat(weights, dim=0)
        try:
            U, S, Vh = linalg.svd(stacked.float(), full_matrices=False)
        except Exception:
            stacked_cpu = stacked.cpu().float()
            del stacked
            U, S, Vh = linalg.svd(stacked_cpu, full_matrices=False)
            del stacked_cpu
        rank = min(Vh.shape[0], Vh.shape[1])
        U_full = U[:, :rank]
        S_full = S[:rank]
        Vh_full = Vh[:rank, :]
        right_factor = torch.diag(S_full) @ Vh_full
        return cls(U_full, right_factor, rank, count, output_width, input_width, dtype, device)

    def bind_quantized_right(self, quantized):
        left = [
            self.left_vectors[index * self.output_width:(index + 1) * self.output_width, :]
            for index in range(self.expert_count)
        ]
        right = quantized["matrix"].to(self.device)
        left = [factor.to(self.device) for factor in left]
        return {
            'SV_shared': right,
            'SV_shared_quant': quantized["quant"],
            'expert_Us': left,
            'rank': self.rank,
            'num_experts': self.expert_count,
            'expert_out_dim': self.output_width,
            'expert_in_dim': self.input_width,
            'original_dtype': self.dtype,
            'original_device': self.device,
        }


@dataclass
class ExpertReconstruction:
    weight: torch.Tensor
    ns_mode: str = "none"
    row_scale: object = None
    column_scale: object = None
    correction_payload: object = None

    def correction_fields(self):
        return {
            "ns_mode": self.ns_mode,
            "ns_vector_row": self.row_scale,
            "ns_vector_col": self.column_scale,
            "c_weight": self.correction_payload,
        }


EXPERT_PAYLOAD_FIELDS = (
    "g_idx", "qweight1", "qweight2_sparse", "qweight2_sparse_prefix",
    "q_gmap", "q_gmap_sparse", "q_gmap_sparse_prefix", "c_map_packed",
    "row_pack1", "row_pack2", "row_pack3", "col_pack3",
    "scales_s_d_c1", "scales_s_d_c2", "scales_s_nd_c1", "scales_s_nd_c2",
    "scales_ns_d_c1", "scales_ns_nd_c1",
)


class SharedRightFactorMixin:
    def _prepare_shared_right_factor(self):
        """Publish row-stacked left factors and one GPTQ-quantized right factor."""
        if not self.weights_by_projection:
            logger.error("Expert weights dictionary is empty")
            return False

        logger.info(f"Starting joint SVD for {self.projection_name}")

        expert_weights = self.weights_by_projection.get(self.projection_name, [])
        if not expert_weights:
            logger.error(f"No expert weights found for {self.projection_name}")
            return False

        if hasattr(self, 'hessian_for_shared_svd') and self.hessian_for_shared_svd is not None:
            H_for_svd = self.hessian_for_shared_svd
            print(f"Using Global Gate Hessian for SVD in {self.layer_name}")
        else:
            H_for_svd = self.H

        decomposition = RowStackDecomposition.compute(expert_weights)
        quantized = self._gptq_quantize_shared_sv(decomposition.right_factor, H_for_svd)
        factors = decomposition.bind_quantized_right(quantized)
        ProjectionFactorStore(self.factorization_cache, self.projection_name).publish(factors)
        return True

    def _quantize_shared_right_factor(self, SV_matrix, H_matrix, blocksize=128, percdamp=0.01, groupsize=-1, actorder=False, static_groups=False):
        actorder = False
        Q, scale, zero, g_idx = self._gptq_quantize_ns(
            SV_matrix,
            H_matrix,
            blocksize=blocksize,
            percdamp=percdamp,
            groupsize=groupsize,
            actorder=actorder,
            static_groups=static_groups,
        )
        return {
            "matrix": Q,
            "quant": build_packed_4bit_payload(Q, scale, zero, g_idx),
        }

    def _reconstruct_factorized_weight(self, expert_idx):
        return ProjectionFactorStore(self.factorization_cache, self.projection_name).reconstruct(expert_idx)

    def _quantize_factorized_weight(self, blocksize, percdamp, orders, num_p):
        store = ProjectionFactorStore(self.factorization_cache, self.projection_name)
        if not store.ensure_ready(self._prepare_shared_right_factor):
            logger.error("SVD decomposition failed")
            return {"error": float('inf')}
        if hasattr(self, "hessian_for_shared_svd"):
            del self.hessian_for_shared_svd

        U_quantized, quant_error, u_payload = self._quantize_expert_u(
            self.expert_index, blocksize, percdamp, orders, num_p
        )

        if U_quantized is None:
            return {"error": float('inf')}

        W_reconstructed = self._reconstruct_factorized_weight(self.expert_index)
        if W_reconstructed is None:
            return {"error": float('inf')}

        reconstruction = self._correct_expert_reconstruction(W_reconstructed)
        self._write_expert_reconstruction(reconstruction.weight)
        return self._assemble_expert_result(quant_error, u_payload, reconstruction)

    def _correct_expert_reconstruction(self, weight):
        reconstruction = ExpertReconstruction(weight)
        if self.ns_rule != "disable":
            try:
                if self.ns_rule == "matrix":
                    if self._uses_shared_gate_up_basis():
                        reconstruction.weight, reconstruction.correction_payload = self._apply_shared_gate_up_null_space_optimization(weight)
                    else:
                        reconstruction.weight, reconstruction.correction_payload = self._apply_null_space_optimization_with_matrix(weight)
                    reconstruction.ns_mode = "matrix"
                elif self.ns_rule == "vector":
                    reconstruction.weight, reconstruction.row_scale, reconstruction.column_scale = self._fit_row_column_correction(weight)
                    reconstruction.ns_mode = "vector"
            except Exception as e:
                logger.warning(f"Null space optimization failed: {e}")
        return reconstruction

    def _write_expert_reconstruction(self, weight):
        if isinstance(self.layer, transformers.Conv1D):
            weight = weight.t()
        self.layer.weight.data = weight.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

    def _assemble_expert_result(self, error, payload, reconstruction):
        if payload is None:
            if self.build_deploy_payload:
                raise RuntimeError("expert_u / mode=2 fallback has been removed; expert quantization now requires expert_compact3 payload generation.")
            result = {"error": error, "mode": "pseudo_expert"}
        else:
            result = {"error": error, "mode": payload["mode"]}
            result.update((key, payload.get(key)) for key in EXPERT_PAYLOAD_FIELDS)
            result["shared_key"] = self.projection_name
        result.update(reconstruction.correction_fields())
        return result
