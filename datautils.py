import random
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, config as datasets_config, load_dataset
from transformers import AutoTokenizer
import os

C4_PREFERRED_OFFLINE_CACHE_ROOT = Path(
    os.environ.get("TRIS_C4_CACHE_ROOT", datasets_config.HF_DATASETS_CACHE)
)


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

'''
Generate tokenizer and return it to preload datasets by converting them to embedded vectors instead of natural words
'''
def get_tokenizer(model, tokenizer_dir=None):
    tokenizer_source = tokenizer_dir if tokenizer_dir else model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    return tokenizer


def _cache_key_component(value):
    if value is None:
        return "none"
    text = str(value)
    return text.replace(os.sep, "__").replace(":", "_")

def get_wikitext2(nsamples, seed, seqlen, model, tokenizer):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')


    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def _dataset_cache_roots(dataset_path):
    dataset_suffix = dataset_path.replace("/", "___")
    candidate_roots = []
    if dataset_path == "allenai/c4":
        preferred_datasets_root = C4_PREFERRED_OFFLINE_CACHE_ROOT / "datasets"
        default_datasets_root = Path(datasets_config.HF_DATASETS_CACHE)
        candidate_roots.extend(
            [
                preferred_datasets_root / dataset_suffix,
                preferred_datasets_root / "json",
                default_datasets_root / dataset_suffix,
                default_datasets_root / "json",
            ]
        )
    else:
        candidate_roots.append(Path(datasets_config.HF_DATASETS_CACHE) / dataset_suffix)

    deduped_roots = []
    seen = set()
    for root in candidate_roots:
        normalized_root = str(root)
        if normalized_root in seen:
            continue
        seen.add(normalized_root)
        deduped_roots.append(root)
    return deduped_roots


def _is_cache_hash_mismatch(error, dataset_path):
    return (
        isinstance(error, ValueError)
        and f"Couldn't find cache for {dataset_path}" in str(error)
    )


def _iter_cached_dataset_dirs(dataset_path):
    def _sorted_dirs(path):
        return sorted(
            (item for item in path.iterdir() if item.is_dir()),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )

    for cache_root in _dataset_cache_roots(dataset_path):
        if not cache_root.exists():
            continue
        for config_dir in _sorted_dirs(cache_root):
            for version_dir in _sorted_dirs(config_dir):
                for hash_dir in _sorted_dirs(version_dir):
                    info_path = hash_dir / "dataset_info.json"
                    if info_path.is_file():
                        yield hash_dir, info_path


def _is_cached_c4_dataset_info(dataset_info):
    if dataset_info.get("dataset_name") == "c4":
        return True

    download_checksums = dataset_info.get("download_checksums", {})
    for source_path in download_checksums:
        source_text = str(source_path)
        if "c4-train.00000-of-01024.json.gz" in source_text:
            return True
        if "c4-validation.00000-of-00008.json.gz" in source_text:
            return True
        if "/datasets/c4/" in source_text.replace("\\", "/"):
            return True

    return False


def _cached_dataset_info_matches(dataset_path, dataset_info):
    if dataset_path == "allenai/c4":
        return _is_cached_c4_dataset_info(dataset_info)
    return True


def _find_cached_arrow_files(hash_dir, split):
    patterns = [
        f"*-{split}-*.arrow",
        f"*-{split}.arrow",
        f"{split}.arrow",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(sorted(hash_dir.glob(pattern)))
    deduped_matches = []
    seen = set()
    for match in matches:
        key = str(match)
        if key in seen:
            continue
        seen.add(key)
        deduped_matches.append(match)
    return deduped_matches


def _load_cached_arrow_split(dataset_path, split):
    for hash_dir, info_path in _iter_cached_dataset_dirs(dataset_path):
        try:
            dataset_info = json.loads(info_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        if not _cached_dataset_info_matches(dataset_path, dataset_info):
            continue

        if split not in dataset_info.get("splits", {}):
            continue

        arrow_files = _find_cached_arrow_files(hash_dir, split)
        if not arrow_files:
            continue

        split_datasets = [Dataset.from_file(str(arrow_file)) for arrow_file in arrow_files]
        if len(split_datasets) == 1:
            return split_datasets[0]
        return concatenate_datasets(split_datasets)

    return None


def _load_dataset_with_cache_fallback(dataset_path, *, split, data_files, num_proc):
    try:
        return load_dataset(
            dataset_path,
            data_files=data_files,
            split=split,
            num_proc=num_proc,
        )
    except ValueError as error:
        if not _is_cache_hash_mismatch(error, dataset_path):
            raise

        cached_dataset = _load_cached_arrow_split(dataset_path, split)
        if cached_dataset is None:
            raise

        warnings.warn(
            f"Falling back to cached Arrow split for {dataset_path}:{split} "
            "because the local datasets cache hash no longer matches the current loader settings.",
            RuntimeWarning,
        )
        return cached_dataset


def get_c4(nsamples, seed, seqlen, model, tokenizer):
    traindata = _load_dataset_with_cache_fallback(
        'allenai/c4',
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train',
        num_proc=48,
    )
    try:
        valdata = _load_dataset_with_cache_fallback(
            'allenai/c4',
            data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
            split='validation',
            num_proc=48,
        )
    except ValueError:
        warnings.warn(
            "C4 validation split is unavailable in the local cache; "
            "falling back to cached train samples for approximate perplexity evaluation.",
            RuntimeWarning,
        )
        valdata = traindata

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model='', tokenizer_dir=None, cache_dir=None):
    model_key = _cache_key_component(model)
    tokenizer_key = _cache_key_component(tokenizer_dir if tokenizer_dir else model)
    cache_root = cache_dir if cache_dir else os.environ.get("TRIS_EVAL_CACHE_DIR", "cache")
    cache_file = os.path.join(cache_root, f'{name}_{nsamples}_{seed}_{seqlen}_{model_key}_{tokenizer_key}.pt')
    try:
        return torch.load(cache_file,weights_only=False)
    except:
        pass

    tokenizer = get_tokenizer(model, tokenizer_dir=tokenizer_dir)
    
    if 'wikitext2' in name:
        loaders= get_wikitext2(nsamples, seed, seqlen, model, tokenizer)
    if 'ptb' in name:
        loaders= get_ptb(nsamples, seed, seqlen, model, tokenizer)
    if 'c4' in name:
        loaders= get_c4(nsamples, seed, seqlen, model, tokenizer)
    directory = os.path.dirname(cache_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(loaders,cache_file)
    return loaders
