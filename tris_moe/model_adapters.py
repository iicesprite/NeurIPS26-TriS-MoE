"""Model projection names, expert grouping, and correction eligibility."""

MIXTRAL_EXPERT_LAYER_MAP = {
    "w1": "gate_proj",
    "w3": "up_proj",
    "w2": "down_proj",
}


def is_shared_expert(layer_name):
    """Check if layer is a shared expert."""
    return 'shared_expert' in layer_name

def is_regular_expert(layer_name):
    """Check if layer is a regular expert."""
    return (
        (('mlp.experts' in layer_name) or ('block_sparse_moe.experts' in layer_name))
        and not is_shared_expert(layer_name)
    )

def is_router_layer(layer_name):
    """Check if layer is the MoE router gate."""
    return (
        ('mlp.gate' in layer_name)
        or ('shared_expert_gate' in layer_name)
        or ('block_sparse_moe.gate' in layer_name)
    )

def is_primary_router_layer(layer_name):
    """Check if layer is the primary MoE router used for expert routing."""
    return ('mlp.gate' in layer_name) or ('block_sparse_moe.gate' in layer_name)

def extract_expert_info(layer_name):
    """Extract expert index and layer type from layer name."""
    if 'mlp.experts' in layer_name:
        parts = layer_name.split('.')
        expert_idx_pos = parts.index('experts') + 1
        expert_idx = int(parts[expert_idx_pos])
        layer_type = '.'.join(parts[expert_idx_pos + 1:])
        return expert_idx, layer_type
    if 'block_sparse_moe.experts' in layer_name:
        parts = layer_name.split('.')
        expert_idx_pos = parts.index('experts') + 1
        expert_idx = int(parts[expert_idx_pos])
        raw_layer_type = '.'.join(parts[expert_idx_pos + 1:])
        layer_type = MIXTRAL_EXPERT_LAYER_MAP.get(raw_layer_type, raw_layer_type)
        return expert_idx, layer_type
    return None, None

def group_layers_by_type(subset):
    """Compatibility view of the ordered Tris weight plan."""
    from .planning import LayerQuantizationPlan

    return LayerQuantizationPlan.from_modules(subset).as_legacy_groups()


def should_use_null_space(layer_name):
    return (
        ('mlp.gate' in layer_name)
        or ('mlp.experts' in layer_name)
        or ('shared_expert_gate' in layer_name)
        or ('block_sparse_moe.gate' in layer_name)
        or ('block_sparse_moe.experts' in layer_name)
    )
