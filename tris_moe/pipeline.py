"""Calibration and quantization of an ordered decoder-layer weight plan."""

from .quantization.factorization import ProjectionFactorStore


class LayerQuantizationSession:
    """Own quantizers and projection caches until a decoder layer is packed."""

    def __init__(self, plan, args, model_type, quantizer_cls, binarizer_cls, logger):
        self.plan = plan
        self.args = args
        self.model_type = model_type
        self.quantizer_cls = quantizer_cls
        self.binarizer_cls = binarizer_cls
        self.logger = logger
        self.quantizers = {}
        self.projection_stores = {}
        try:
            self._initialize_quantizers()
        except BaseException:
            self.release()
            raise

    def release(self):
        """Free unfinished quantizers and drop all session-owned cache references."""
        try:
            for quantizer in self.quantizers.values():
                quantizer.free()
        finally:
            self.quantizers.clear()
            for store in self.projection_stores.values():
                store.release()
            self.projection_stores.clear()

    def _make_quantizer(self, target, **projection_inputs):
        binary_quantizer = self.binarizer_cls(target.module.weight, groupsize=self.args.blocksize)
        return self.quantizer_cls(
            target.module,
            binary_quantizer,
            disable_gptq=self.args.disable_gptq,
            build_deploy_payload=self.args.real_quant,
            is_expert_layer=target.role == "factorized",
            **projection_inputs,
            layer_name=target.name,
            idx_rule=self.args.idx_rule,
            ns_rule=self.args.ns_rule,
            shared_gate_up_basis=self.args.shared_gate_up_basis,
            model_type=self.model_type,
        )

    def _initialize_quantizers(self):
        for target in self.plan.direct_targets:
            self.quantizers[target.name] = self._make_quantizer(target)

        # Snapshot every projection before creating any expert quantizer.
        weights = {}
        for group in self.plan.projections:
            weights[group.name] = group.copy_weights()
            self.logger.info(f"  Prepared joint decomposition for {group.name}: {len(weights[group.name])} experts")

        for group in self.plan.projections:
            store = ProjectionFactorStore({}, group.name)
            self.projection_stores[group.name] = store
            self.logger.info(f"  Expert NS rule for {group.name}: {self.args.ns_rule}")
            for target in group.ordered_members():
                self.quantizers[target.name] = self._make_quantizer(
                    target,
                    weights_by_projection=weights,
                    projection_name=group.name,
                    expert_index=target.expert_index,
                    factorization_cache=store.entries,
                )

    def attach_calibration_hooks(self, modules):
        def capture(name):
            def hook(_, inputs, output):
                self.quantizers[name].add_batch(inputs[0].data, output.data)
            return hook

        handles = []
        try:
            for name in self.quantizers:
                handles.append(modules[name].register_forward_hook(capture(name)))
        except BaseException:
            for handle in handles:
                handle.remove()
            raise
        return handles

    def bind_projection_metrics(self, is_primary_router):
        router_hessian = None
        router_second_moment = None
        for target in self.plan.direct_targets:
            if target.role == "direct" and is_primary_router(target.name):
                quantizer = self.quantizers[target.name]
                router_hessian = quantizer.H.clone()
                router_second_moment = quantizer.input_second_moment.clone() if quantizer.input_second_moment is not None else None
                self.logger.info(f"    [Config] Captured Hessian from {target.name} for shared SVD.")
                break

        if router_hessian is None:
            self.logger.warning("    [Config] Warning: primary MoE router not found. gate_proj/up_proj will fallback to local H.")
        elif self.args.shared_gate_up_basis and router_second_moment is None:
            self.logger.warning("    [Config] Warning: primary router input_second_moment is unavailable. Shared gate/up null-space basis will fallback to local basis.")

        for group in self.plan.projections:
            if 'gate_proj' in group.name or 'up_proj' in group.name:
                if router_hessian is not None:
                    self._bind_router_metric(group, router_hessian, router_second_moment)
            elif 'down_proj' in group.name:
                self._bind_mean_expert_metric(group)

    def _bind_router_metric(self, group, hessian, second_moment):
        count = 0
        # Preserve discovery order here, including independently cloned metrics.
        for target in group.members:
            if target.name in self.quantizers:
                quantizer = self.quantizers[target.name]
                quantizer.hessian_for_shared_svd = hessian.clone()
                if self.args.shared_gate_up_basis and second_moment is not None:
                    quantizer.hessian_for_shared_ns = hessian.clone()
                    quantizer.shared_input_second_moment = second_moment.clone()
                count += 1
        self.logger.info(f"      -> Injected Gate H to {count} experts in {group.name}")

    def _bind_mean_expert_metric(self, group):
        accumulator = None
        count = 0
        for target in group.ordered_members():
            if target.name in self.quantizers:
                hessian = self.quantizers[target.name].H
                if accumulator is None:
                    accumulator = hessian.clone()
                else:
                    accumulator += hessian
                count += 1
        if count > 0:
            mean = accumulator / count
            for target in group.ordered_members():
                if target.name in self.quantizers:
                    self.quantizers[target.name].hessian_for_shared_svd = mean

    def quantize(self, layer_index, is_router, make_payload, bank_key, to_cpu):
        layer_payloads = {}
        shared_payloads = {}
        for target in self.plan.direct_targets:
            self._quantize_target(target, layer_index, is_router, make_payload, bank_key, layer_payloads)

        for group in self.plan.projections:
            self.logger.info(f"    Joint quantizing {group.name} expert group...")
            for target in group.ordered_members():
                self._quantize_target(target, layer_index, is_router, make_payload, bank_key, layer_payloads)
            store = self.projection_stores[group.name]
            if self.args.real_quant:
                bank = store.export_bank(to_cpu)
                if bank is not None:
                    shared_payloads[bank_key(layer_index, group.name)] = bank
            store.release()
            self.logger.info(f"    {group.name} expert group quantization completed")
        return layer_payloads, shared_payloads

    def _quantize_target(self, target, layer_index, is_router, make_payload, bank_key, payloads):
        if target.name not in self.quantizers:
            return
        quantizer = self.quantizers[target.name]
        factorized = target.role == "factorized"
        if not factorized and not self.args.quant_router and is_router(target.name):
            self.logger.info(f"    Skipping router quantization: Layer {layer_index} - {target.name}")
            quantizer.free()
            del self.quantizers[target.name]
            return

        indent = "      " if factorized else "    "
        self.logger.info(f"{indent}Quantizing: Layer {layer_index} - {target.name}")
        result = quantizer.fasterquant(
            percdamp=self.args.percdamp,
            blocksize=self.args.blocksize,
            num_p=1,
        )
        self.logger.info(f"{indent}  Quantization error: {result['error']:.6f}")
        if self.args.real_quant:
            key = f'model.layers.{layer_index}.{target.name}'
            if factorized:
                payloads[key] = make_payload(result, extra={'shared_key': bank_key(layer_index, target.projection)})
            else:
                payloads[key] = make_payload(result)
        quantizer.free()
        del self.quantizers[target.name]
