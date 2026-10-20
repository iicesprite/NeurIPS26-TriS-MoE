"""Quantizer API; numerical stages are implemented in tris_moe.quantization."""

import math
import time
import torch
import torch.nn as nn
from null_space_utils import find_max_curvature_index, find_max_kneedle_index,find_max_fit_index,find_max_fit_nohead_index,fast_find_energy_drop_index
import transformers
from utils.structure import structural_guassian_distribution_multip_alternating_group_x
from utils.pack_utils import (
    append_exact_zero_row_to_packed_4bit_payload,
    append_zero_column_to_packed_4bit_payload,
    build_compact3_packed_payload,
    build_packed_4bit_payload,
    pad_nullspace_payloads_even_rank,
)
from gptq.quant import *
import logging
import torch.linalg as linalg
from tris_moe.model_adapters import should_use_null_space
from tris_moe.quantization.calibration import CalibrationStatisticsMixin
from tris_moe.quantization.factorization import SharedRightFactorMixin
from tris_moe.quantization.correction import SpectralCorrectionMixin
from tris_moe.quantization.blockwise import BlockwiseQuantizationMixin
from tris_moe.quantization.auxiliary import AuxiliaryMatrixQuantizationMixin

logger = logging.getLogger()
DEBUG = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _cuda_synchronize_for(device):
    if isinstance(device, torch.Tensor):
        device = device.device
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.synchronize(device)


def _cuda_empty_cache_for(device):
    if isinstance(device, torch.Tensor):
        device = device.device
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    with torch.cuda.device(device):
        torch.cuda.empty_cache()


class TriSGPTQ(
    CalibrationStatisticsMixin,
    SharedRightFactorMixin,
    SpectralCorrectionMixin,
    BlockwiseQuantizationMixin,
    AuxiliaryMatrixQuantizationMixin,
):
    """Public quantizer API with separate calibration, factorization, and correction stages."""

    def __init__(
        self,
        layer,
        braq_quantizer,
        disable_gptq=False,
        build_deploy_payload=False,
        # MoE expert parameters
        is_expert_layer=False,
        weights_by_projection=None,
        projection_name=None,
        expert_index=None,
        factorization_cache=None,
        layer_name=None,
        idx_rule=None,
        ns_rule=None,
        shared_gate_up_basis=False,
        model_type=None,
    ):
        self.layer = layer
        self.dev = layer.weight.device
        self.layer_name = layer_name
        self.idx_rule=idx_rule
        self.ns_rule=ns_rule
        self.shared_gate_up_basis = shared_gate_up_basis
        self.model_type = str(model_type or "").lower()

        W = layer.weight.data.clone()
        if isinstance(layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(layer, transformers.Conv1D):
            W = W.t()
        self.rows, self.columns = W.shape

        self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float32)
        self.nsamples = 0
        self.input_second_moment = None

        self.braq_quantizer = braq_quantizer
        self.disable_gptq = disable_gptq
        self.build_deploy_payload = build_deploy_payload

        self.is_expert_layer = is_expert_layer
        self.weights_by_projection = weights_by_projection
        self.projection_name = projection_name
        self.expert_index = expert_index
        self.factorization_cache = factorization_cache

        self.shared_input_second_moment = None
        self.hessian_for_shared_ns = None

    def _uses_shared_gate_up_basis(self):
        return (
            self.shared_gate_up_basis
            and self.is_expert_layer
            and self.projection_name is not None
            and ("gate_proj" in self.projection_name or "up_proj" in self.projection_name)
        )


    def _select_ns_index(self, s):
        if s.numel() == 0:
            return 0

        index = 0
        if self.idx_rule == "fast":
            threshold = 0.1
            index = fast_find_energy_drop_index(s, threshold=threshold)
        elif self.idx_rule == "curvature":
            index = find_max_curvature_index(s, self.layer_name)
        elif self.idx_rule == "kneedle":
            index = find_max_kneedle_index(s, self.layer_name)
        elif self.idx_rule == "fit":
            index = find_max_fit_index(s, self.layer_name)
        elif self.idx_rule == "fit_nohead":
            index = find_max_fit_nohead_index(s, self.layer_name)

        logger.info(f"{self.layer_name} ns_index: {index}")

        if index is None:
            print("index is none")
            return 0

        return int(max(0, min(index, s.numel() - 1)))

    def _build_null_space_eigh_matrix(self, avg_x, context):
        """Compatibility entrypoint; implementation lives in the TriS quantization package."""
        return self._prepare_symmetric_second_moment(avg_x, context)

    def _build_shared_gate_up_ns_basis(self):
        """Compatibility entrypoint; implementation lives in the TriS quantization package."""
        return self._prepare_shared_correction_basis()

    def _apply_shared_gate_up_null_space_optimization(self, W_reconstructed):
        """Compatibility entrypoint; implementation lives in the TriS quantization package."""
        return self._apply_shared_basis_correction(W_reconstructed)

    def _build_compact3_packed_payload(self, *args, **kwargs):
        return build_compact3_packed_payload(device=self.dev, *args, **kwargs)

    def _build_packed_4bit_payload(self, matrix, scale, zero, g_idx):
        return build_packed_4bit_payload(matrix, scale, zero, g_idx)

    def _append_zero_column_to_packed_4bit_payload(self, payload):
        return append_zero_column_to_packed_4bit_payload(payload)

    def _append_exact_zero_row_to_packed_4bit_payload(self, payload):
        return append_exact_zero_row_to_packed_4bit_payload(payload)

    def _pad_nullspace_payloads_even_rank(self, c1_payload, c2_payload):
        return pad_nullspace_payloads_even_rank(c1_payload, c2_payload)

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        return self._accumulate_input_statistics(inp)

    def _gptq_quantize_shared_sv(self, SV_matrix, H_matrix, blocksize=128, percdamp=0.01, groupsize=-1, actorder=False, static_groups=False):
        """Compatibility entrypoint; implementation lives in the TriS quantization package."""
        return self._quantize_shared_right_factor(SV_matrix, H_matrix, blocksize, percdamp, groupsize, actorder, static_groups)

    def _gptq_quantize_ns(self, ns_matrix, H_matrix, blocksize=128, percdamp=0.01, groupsize=-1, actorder=False, static_groups=False):
        """Compatibility entrypoint; implementation lives in the TriS quantization package."""
        return self._quantize_auxiliary_matrix(ns_matrix, H_matrix, blocksize, percdamp, groupsize, actorder, static_groups)

    def _quantize_expert_u(self, expert_idx, blocksize, percdamp, orders, num_p):
        """Compatibility entrypoint; implementation lives in the TriS quantization package."""
        return self._quantize_expert_left_factor(expert_idx, blocksize, percdamp, orders, num_p)

    def _apply_null_space_optimization_with_matrix(self, W_reconstructed):
        """Compatibility entrypoint; implementation lives in the TriS quantization package."""
        return self._apply_low_rank_correction(W_reconstructed)

    def fasterquant(self, blocksize=128, percdamp=0.01, orders=(1, 1, 2), num_p=1):
        if self.is_expert_layer:
            return self._quantize_factorized_weight(blocksize, percdamp, orders, num_p)
        return self._quantize_dense_weight(blocksize, percdamp, orders, num_p)

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        _cuda_empty_cache_for(self.dev)

    def _synchronize_device(self, device):
        return _cuda_synchronize_for(device)

    def _empty_device_cache(self, device):
        return _cuda_empty_cache_for(device)

    def _correction_applies(self, layer_name):
        return should_use_null_space(layer_name)
