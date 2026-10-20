import time

import torch
import torch.nn as nn

import logging
logger = logging.getLogger()


def _normalize_device(device):
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _empty_cuda_cache_for(device):
    device = _normalize_device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

@torch.no_grad()
def get_loss(model, dataloader, dev,nsamples,seqlen):
    print("Get global loss ...")
    dev = _normalize_device(dev)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, 'rotary_emb') and model.model.rotary_emb is not None:
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {"i": 0, "inps": inps, "kwargs": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["kwargs"] = kwargs
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    _empty_cuda_cache_for(dev)

    outs = torch.zeros_like(inps)

    layer_kwargs = cache["kwargs"]

    for k, v in layer_kwargs.items():
        if isinstance(v, torch.Tensor):
            layer_kwargs[k] = v.to(dev)

    for i in range(len(layers)):
        # print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), ** layer_kwargs)[0]
        layers[i] = layer.cpu()
        del layer
        _empty_cuda_cache_for(dev)
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    losses = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = dataloader[i][0].to(dev)  
        shift_labels = labels[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        losses.append(loss)
    model.config.use_cache = use_cache
    del cache,inps,outs
    return losses


@torch.no_grad()
def moe_eval(model, testenc, dev, dataset: str):
    print("Evaluating ...")
    dev = _normalize_device(dev)

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {"i": 0, "inps": inps, "kwargs": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["kwargs"] = kwargs
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    _empty_cuda_cache_for(dev)

    outs = torch.zeros_like(inps)

    layer_kwargs = cache["kwargs"]

    for k, v in layer_kwargs.items():
        if isinstance(v, torch.Tensor):
            layer_kwargs[k] = v.to(dev)

    for i in range(len(layers)):
        # print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), ** layer_kwargs)[0]
        layers[i] = layer.cpu()
        del layer
        _empty_cuda_cache_for(dev)
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    logger.info(f"{dataset}/Perplexity: {ppl.item():3f}")

    model.config.use_cache = use_cache

