import os
import time
import torch
import torch.nn as nn
import logging
from trisgptq import TriSGPTQ
from binary import Binarization
from modelutils import find_layers
from datautils import  get_loaders
from transformers import AutoModelForCausalLM,AutoConfig
import sys
from tris_moe.model_adapters import (
    MIXTRAL_EXPERT_LAYER_MAP,
    is_shared_expert,
    is_regular_expert,
    is_router_layer,
    is_primary_router_layer,
    extract_expert_info,
    group_layers_by_type,
)
from tris_moe.planning import LayerQuantizationPlan
from tris_moe.pipeline import LayerQuantizationSession
from tris_moe.payloads import (
    safe_cpu,
    build_quantizer_payload,
    pack_quantizers,
    shared_bank_key,
)

logger = logging.getLogger(__name__)


DEFAULT_LONG_MOE_SEQLEN = 4096


def normalize_device(device):
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def set_cuda_device_if_needed(device):
    device = normalize_device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def empty_cuda_cache_for(device):
    device = normalize_device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    with torch.cuda.device(device):
        torch.cuda.empty_cache()


def ensure_tris_moe_auto_classes_for_model_load():
    autogptq_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AutoGPTQ")
    if autogptq_path not in sys.path:
        sys.path.append(autogptq_path)
    try:
        from auto_gptq.modeling._utils import ensure_shared_olmoe_auto_classes_registered

        ensure_shared_olmoe_auto_classes_registered()
    except Exception as exc:
        logger.warning("Failed to register AutoGPTQ Tris-MoE classes before model load: %r", exc)


def setup_logging(args):
    """Setup logging configuration"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    model_name = os.path.basename(os.path.normpath(args.model))
    
    # args_str = f"dataset-{args.dataset}_blk-{args.blocksize}_ns-{args.nsamples}"
    # if args.disable_gptq:
    #     args_str += "_nogptq"
        
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    os.makedirs("runtime_log", exist_ok=True)
    log_file = f"runtime_log/{model_name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_model(model):
    import torch

    def skip(*args, **kwargs):
        pass

    ensure_tris_moe_auto_classes_for_model_load()

    try:
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    except Exception as e:
        print(f"Config load failed, falling back to default settings. Error: {e}")
        config = None
    model_type = getattr(config, "model_type", "").lower() if config else ""

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip 
    model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype="auto",  
            trust_remote_code=True  
        )
    model.seqlen = 2048
    if "qwen" in model_type:
        model.seqlen = DEFAULT_LONG_MOE_SEQLEN
    elif "olmoe" in model_type:
        model.seqlen = DEFAULT_LONG_MOE_SEQLEN
    elif "mixtral" in model_type:
        model.seqlen = min(
            int(getattr(config, "max_position_embeddings", DEFAULT_LONG_MOE_SEQLEN)),
            DEFAULT_LONG_MOE_SEQLEN,
        )
    elif "deepseek" in model_type:
        model.seqlen = DEFAULT_LONG_MOE_SEQLEN
    else:
        if hasattr(config, "max_position_embeddings"):
            model.seqlen = config.max_position_embeddings
        elif hasattr(config, "max_sequence_length"): 
            model.seqlen = config.max_sequence_length
        elif hasattr(config, "n_positions"):
            model.seqlen = config.n_positions

    return model


@torch.no_grad()
def quant_sequential(model, dataloader, device, args):
    """Sequential quantization for MoE models."""
    logger.info("Starting MoE quantization...")
    device = normalize_device(device)
    model_type = getattr(model.config, "model_type", "")

    for name, module in model.named_modules():
        module.global_name = args.model + name

    use_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.norm = model.model.norm.to(device)
        if hasattr(model.model, 'rotary_emb') and model.model.rotary_emb is not None:
            model.model.rotary_emb = model.model.rotary_emb.to(device)

        layers[0] = layers[0].to(device)
        dtype = next(iter(model.parameters())).dtype
        cpu_pin = device.type == "cuda" and torch.cuda.is_available()
        inps = torch.zeros(
            (args.nsamples, model.seqlen, model.config.hidden_size),
            dtype=dtype,
            device="cpu",
            pin_memory=cpu_pin,
        )
        cache = {"i": 0, "inps": inps, "kwargs": None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache["i"]].copy_(inp.squeeze(0).to(device="cpu", dtype=dtype))
                cache["kwargs"] = kwargs
                cache["i"] += 1
                cache["attention_mask"] = kwargs["attention_mask"]
                raise ValueError

        layers[0] = Catcher(layers[0])
        try:
            for batch in dataloader:
                try:
                    model(batch[0].to(device))
                except ValueError:
                    pass
        finally:
            layers[0] = layers[0].module
            layers[0] = layers[0].cpu()
            model.model.embed_tokens = model.model.embed_tokens.cpu()
            model.model.norm = model.model.norm.cpu()
            if hasattr(model.model, 'rotary_emb') and model.model.rotary_emb is not None:
                model.model.rotary_emb = model.model.rotary_emb.cpu()
            empty_cuda_cache_for(device)

        outs = torch.zeros_like(inps)
        quantizers = {}
        shared_banks = {}
        if args.real_quant:
            logger.info("Real-quant mode: pack each layer in place and skip the final full-model repack.")
        logger.info("Ready.")

        layer_kwargs = cache["kwargs"]
        for k, v in layer_kwargs.items():
            if isinstance(v, torch.Tensor):
                layer_kwargs[k] = v.to(device)

        # Quantize layers sequentially
        for i in range(len(layers)):
            logger.info(f"\nProcessing layer {i}/{len(layers)-1}")
            layer = layers[i].to(device)
            subset = find_layers(layer)

            plan = LayerQuantizationPlan.from_modules(subset)
            router_count = sum(1 for name in subset if is_router_layer(name))
            logger.info(f"  Normal matrix: {sum(t.role == 'direct' for t in plan.direct_targets)}")
            logger.info(f"  Shared experts: {sum(t.role == 'shared' for t in plan.direct_targets)}")
            logger.info(f"  Regular experts: {sum(len(group.members) for group in plan.projections)}")
            if not args.quant_router and router_count:
                logger.info(f"  Router layers kept in original precision: {router_count}")

            session = LayerQuantizationSession(plan, args, model_type, TriSGPTQ, Binarization, logger)
            try:
                handles = session.attach_calibration_hooks(subset)
                try:
                    for j in range(args.nsamples):
                        current_inp = inps[j].unsqueeze(0).to(device=device, dtype=dtype, non_blocking=cpu_pin)
                        current_out = layer(current_inp, **layer_kwargs)[0]
                        outs[j].copy_(current_out.squeeze(0).to(device="cpu", dtype=dtype))
                finally:
                    for h in handles:
                        h.remove()

                session.bind_projection_metrics(is_primary_router_layer)
                layer_payloads, layer_shared_payloads = session.quantize(
                    i, is_router_layer, build_quantizer_payload, shared_bank_key, safe_cpu,
                )

                for j in range(args.nsamples):
                    current_inp = inps[j].unsqueeze(0).to(device=device, dtype=dtype, non_blocking=cpu_pin)
                    current_out = layer(current_inp, **layer_kwargs)[0]
                    outs[j].copy_(current_out.squeeze(0).to(device="cpu", dtype=dtype))

                layers[i] = layer.cpu()
                if args.real_quant:
                    pack_quantized_layer_inplace(model, i, layer_payloads, layer_shared_payloads, bits=1, group_size=args.blocksize)
                del layer_payloads
                del layer_shared_payloads
            finally:
                session.release()
            del layer
            del session
            del plan
            empty_cuda_cache_for(device)

            inps, outs = outs, inps
    finally:
        model.config.use_cache = use_cache

    logger.info("MoE quantization completed")
    return {"layers": quantizers, "shared_banks": shared_banks}


def save_quantized_model(model, save_path):
    """Save quantized model to specified path."""
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path, use_safetensors=True)
    logger.info("Model saved successfully.")


def prepare_bucket_deploy_state(autogptq_model):
    logger.info("Bucket deploy prebuild disabled; saving canonical c_map/g_map structured weights.")


def get_gptq_wrapper_cls_for_model(model):
    from auto_gptq.modeling import (
        MixtralGPTQForCausalLM,
        OlmoeGPTQForCausalLM,
        Qwen2MoeGPTQForCausalLM,
        Qwen3_5MoeGPTQForCausalLM,
    )

    model_type = getattr(model.config, "model_type", "").lower()
    if "qwen3_5_moe_text" in model_type:
        return Qwen3_5MoeGPTQForCausalLM
    if "olmoe" in model_type:
        return OlmoeGPTQForCausalLM
    if "mixtral" in model_type:
        return MixtralGPTQForCausalLM
    if "qwen2_moe" in model_type:
        return Qwen2MoeGPTQForCausalLM
    raise NotImplementedError(f"Real quant save is not implemented for model type: {model.config.model_type}")


def pack_quantized_layer_inplace(model, layer_idx, layer_payloads, layer_shared_banks, bits, group_size):
    import sys, os
    sys.path.append(os.path.join(os.getcwd(), './AutoGPTQ'))
    from auto_gptq.modeling._utils import (
        bind_TriS_shared_banks,
        ensure_TriS_shared_banks,
        find_layers as autogptq_find_layers,
        make_quant as autogptq_make_quant,
    )
    from auto_gptq.nn_modules.qlinear.qlinear_cuda import QuantLinear

    if not layer_payloads and not layer_shared_banks:
        return

    ensure_TriS_shared_banks(model)
    layer = model.model.layers[layer_idx]
    float_layers = autogptq_find_layers(layer)

    prefix = f'model.layers.{layer_idx}.'
    local_payloads = {}
    for full_name, payload in layer_payloads.items():
        if full_name.startswith(prefix):
            local_payloads[full_name[len(prefix):]] = payload

    if local_payloads:
        autogptq_make_quant(
            layer,
            local_payloads,
            bits,
            group_size,
            use_triton=False,
            disable_exllama=False,
            disable_exllamav2=True,
            use_cuda_fp16=True,
            desc_act=False,
            use_tritonv2=False,
        )
        bind_TriS_shared_banks(model)
        qlayers = autogptq_find_layers(layer, [QuantLinear])
        for name, payload in local_payloads.items():
            qlayer = qlayers[name]
            src_layer = float_layers[name].to('cpu')
            qlayer.to('cpu')
            payload_for_pack = dict(payload)
            c_weight = payload_for_pack.get('c_weight')
            if isinstance(c_weight, dict) and c_weight.get('shared_c2'):
                shared_payload = layer_shared_banks.get(payload_for_pack.get('shared_key'))
                shared_ns_c2 = shared_payload.get('shared_ns_c2') if shared_payload else None
                if shared_ns_c2 is None:
                    logger.warning(f"Missing shared_ns_c2 for {prefix}{name}; disabling matrix null-space payload during pack.")
                    payload_for_pack['ns_mode'] = None
                    payload_for_pack['c_weight'] = None
            if payload_for_pack.get('mode') in {'structured', 'compact3'}:
                qlayer.pack_structured(src_layer, payload_for_pack)
            elif payload_for_pack.get('mode') in {'expert', 'expert_structured', 'expert_compact3'}:
                qlayer.pack_expert(src_layer, payload_for_pack)
            else:
                qlayer.pack_dense(src_layer, payload_for_pack)

    for shared_name, payload in layer_shared_banks.items():
        if not payload:
            continue
        module = model.get_submodule(shared_name)
        module.set_shared_sv(payload['shared_sv'])
        module.set_shared_ns_c2(payload.get('shared_ns_c2'))


def save_packed_quantized_model(model, save_path, bits, group_size, extra_quant_config=None, max_shard_size="5GB"):
    import sys, os
    sys.path.append(os.path.join(os.getcwd(), './AutoGPTQ'))
    from auto_gptq import BaseQuantizeConfig

    quant_config_kwargs = {
        'bits': bits,
        'group_size': group_size,
        'desc_act': False,
    }
    if extra_quant_config:
        quant_config_kwargs.update(extra_quant_config)
    quantize_config = BaseQuantizeConfig(**quant_config_kwargs)

    wrapper_cls = get_gptq_wrapper_cls_for_model(model)
    wrapper = wrapper_cls(model, quantized=True, quantize_config=quantize_config)
    wrapper.save_quantized(save_path, use_safetensors=True, max_shard_size=max_shard_size)


def pack_model(model, save_path, bits, group_size, quantizers, extra_quant_config=None):
    import sys, os
    #os.environ["PYTHONPATH"]="$PYTHONPATH:/path/to/AutoGPTQ"
    sys.path.append(os.path.join(os.getcwd(), './AutoGPTQ'))
    from auto_gptq import BaseQuantizeConfig

    quant_config_kwargs = {
        'bits': bits,
        'group_size': group_size,
        'desc_act': False,
    }
    if extra_quant_config:
        quant_config_kwargs.update(extra_quant_config)
    quantize_config = BaseQuantizeConfig(**quant_config_kwargs)

    wrapper_cls = get_gptq_wrapper_cls_for_model(model)
    model = wrapper_cls(model, quantized=False, quantize_config=quantize_config)
    model.quantize([], layer_quant_params=quantizers)
    model.save_quantized(save_path, use_safetensors=True)

def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TriSMoE")

    # Basic arguments
    parser.add_argument("model", type=str, help="Model path or HuggingFace model name")
    parser.add_argument("dataset", type=str, choices=["wikitext2", "ptb", "c4"], help="Calibration dataset")

    # Quantization parameters
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--percdamp", type=float, default=0.01, help="Hessian damping percentage")
    parser.add_argument("--blocksize", type=int, default=128, help="Block size for quantization")

    #GPTQ options
    parser.add_argument("--disable_gptq", action="store_true", help="Disable GPTQ for quantization")
    parser.add_argument(
        "--quant_router",
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help="Whether to quantize MoE router layers. Set --quant_router false to keep router weights in original precision and skip qlinear replacement.",
    )

    # Save and evaluation
    parser.add_argument("--save", type=str, default=None, help="Path to save quantized model")
    parser.add_argument(
        "--save_max_shard_size",
        type=str,
        default="5GB",
        help="Maximum size per safetensors shard when saving quantized checkpoints, e.g. 5GB or 2GB.",
    )
    parser.add_argument(
        "--real_quant",
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help="Whether to save a real quantized deployment model. Set --real_quant false to save a pseudo-quantized fp model.",
    )
    parser.add_argument("--eval", action="store_true", help="Evaluate model after quantization")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for quantization")
    parser.add_argument("--seqlen", type=int, default=None, help="Override calibration/evaluation sequence length.")

    parser.add_argument("--idx_rule", type=str, choices=["fast", "kneedle", "fit", "fit_nohead", "curvature"], help="algorithm to obtain index k")
    parser.add_argument("--ns_rule", type=str, choices=["vector", "matrix", "disable"], help="null space strategy")
    parser.add_argument("--shared_gate_up_basis", action="store_true", help="Enable a shared global null-space basis for gate_proj/up_proj experts while keeping the original local null-space algorithm as the default.")
    args = parser.parse_args()
    args.device = set_cuda_device_if_needed(args.device)

    setup_logging(args)

    if args.shared_gate_up_basis and args.ns_rule != "matrix":
        logger.warning(
            "shared_gate_up_basis is enabled but ns_rule is not matrix; "
            "the shared gate/up null-space basis path will be inactive."
        )

    logger.info("=" * 50)
    logger.info(f"Executing Script : {os.path.basename(sys.argv[0])}")
    logger.info("Run Configurations:")
    for key, value in vars(args).items():
        logger.info(f"  {key:<20}: {value}")
    logger.info("=" * 50)

    # Load model and data
    model = get_model(args.model)
    if args.seqlen is not None:
        if args.seqlen <= 0:
            raise ValueError(f"--seqlen must be positive, got {args.seqlen}")
        logger.info(f"Overriding model.seqlen: {model.seqlen} -> {args.seqlen}")
        model.seqlen = args.seqlen
    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen
    )

    # Execute quantization
    start_time = time.time()
    quantizers = quant_sequential(model, dataloader, args.device, args)
    end_time = time.time()
    
    logger.info(f"Time elapsed: {end_time - start_time:.2f}s")

    # Evaluate model
    if args.eval:
        for dataset in ["wikitext2"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, seqlen=model.seqlen, model=args.model
            )
            logger.info(f"Evaluating on {dataset}")
            
            if any(model_name in args.model.lower() for model_name in ["qwen", "deepseek", "olmoe", "mixtral"]):
                from eval_ppl_utils import moe_eval
                moe_eval(model, testloader, args.device, dataset)

        # Save model
    if args.save:
        if args.real_quant:
            logger.info("Saving real quantized model...")
            extra_quant_config = {
                'blocksize': args.blocksize,
                'ns_rule': args.ns_rule,
                'quant_router': bool(args.quant_router),
                'shared_gate_up_basis': bool(args.shared_gate_up_basis),
                'shared_sv_group_size': -1,
                'shared_sv_bits': 4,
                'ns_group_size': -1,
                'ns_bits': 4,
            }
            save_packed_quantized_model(
                model,
                args.save,
                bits=1,
                group_size=args.blocksize,
                extra_quant_config=extra_quant_config,
                max_shard_size=args.save_max_shard_size,
            )
        else:
            logger.info("Saving pseudo-quantized model...")
            save_quantized_model(model, args.save)
