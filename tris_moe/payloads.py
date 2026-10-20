"""Stable tensor dictionaries and shared-bank keys used by checkpoints."""

import torch


def safe_cpu(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu()
    if isinstance(tensor, list):
        return [safe_cpu(item) for item in tensor]
    if isinstance(tensor, tuple):
        return tuple(safe_cpu(item) for item in tensor)
    if isinstance(tensor, dict):
        return {k: safe_cpu(v) for k, v in tensor.items()}
    return tensor

def build_quantizer_payload(results, extra=None):
    payload = {
        'mode': results.get('mode', 'dense'),
        'g_idx': safe_cpu(results.get('g_idx')),
        'qweight1': safe_cpu(results.get('qweight1')),
        'qweight2': safe_cpu(results.get('qweight2')),
        'qweight2_sparse': safe_cpu(results.get('qweight2_sparse')),
        'qweight2_sparse_prefix': safe_cpu(results.get('qweight2_sparse_prefix')),
        'q_gmap': safe_cpu(results.get('q_gmap')),
        'q_gmap_sparse': safe_cpu(results.get('q_gmap_sparse')),
        'q_gmap_sparse_prefix': safe_cpu(results.get('q_gmap_sparse_prefix')),
        'c_map_packed': safe_cpu(results.get('c_map_packed')),
        'row_pack1': safe_cpu(results.get('row_pack1')),
        'row_pack2': safe_cpu(results.get('row_pack2')),
        'col_pack3': safe_cpu(results.get('col_pack3')),
        'row_pack3': safe_cpu(results.get('row_pack3')),

        # Scale tensors.
        'scales_s_d_c1': safe_cpu(results.get('scales_s_d_c1')),
        'scales_s_d_c2': safe_cpu(results.get('scales_s_d_c2')),
        'scales_s_nd_c1': safe_cpu(results.get('scales_s_nd_c1')),
        'scales_s_nd_c2': safe_cpu(results.get('scales_s_nd_c2')),
        'scales_ns_d_c1': safe_cpu(results.get('scales_ns_d_c1')),
        'scales_ns_nd_c1': safe_cpu(results.get('scales_ns_nd_c1')),
        'ns_mode': results.get('ns_mode'),
        'ns_vector_row': safe_cpu(results.get('ns_vector_row')),
        'ns_vector_col': safe_cpu(results.get('ns_vector_col')),
        'c_weight': safe_cpu(results.get('c_weight')),
    }
    if extra:
        payload.update(extra)
    return payload

def pack_quantizers(quantizers, results, layer_key):
    quantizers[layer_key] = build_quantizer_payload(results)
    return quantizers

def shared_bank_key(layer_idx, layer_type):
    return f"model.layers.{layer_idx}.TriS_shared_{layer_type.replace('.', '_')}"
