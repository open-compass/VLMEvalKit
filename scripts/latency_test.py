#!/usr/bin/env python3
"""
Latency Test Script for (VT, VTVT) × (CoT, noCoT) combinations.
Tests TTFT, TTPT, and total latency on a configurable sample of each dataset.

Usage:
    conda activate vlmeval
    python scripts/latency_test.py --config scripts/configs/latency_qwen25vl_3b.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None


def load_yaml_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if yaml is not None:
            return yaml.safe_load(f)
        else:
            return json.load(f)


def sample_dataset(dataset, fraction: float = 0.01, random_state: int = 42):
    """Sample a fraction of the dataset for latency testing."""
    data = dataset.data
    if fraction >= 1.0:
        return dataset
    n_samples = max(1, int(len(data) * fraction))
    sampled = data.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
    dataset.data = sampled
    print(f"[sampling] {dataset.dataset_name}: {sampled.shape[0]}/{len(data)} samples ({fraction*100:.1f}%)", flush=True)
    return dataset


def build_latency_dataset(dataset_name: str, fraction: float = 0.01, allowlist: list[int] | None = None):
    """Build a dataset and sample a fraction.

    If allowlist is provided, filter by the given indices instead of random sampling.
    This enables exact sample alignment with smoke_test.
    """
    from vlmeval.dataset import build_dataset

    ds = build_dataset(dataset_name)
    if ds is None:
        raise ValueError(f"Failed to build dataset: {dataset_name}")

    if allowlist is not None:
        # Filter by allowlist (aligned with smoke_test's _filter_data_by_allowlist)
        data = ds.data
        allowlist_set = set(allowlist)
        data_indices = list(data['index'])
        mask = [idx in allowlist_set for idx in data_indices]
        ds.data = data[mask].reset_index(drop=True)
        print(f"[allowlist] {dataset_name}: {ds.data.shape[0]}/{len(data)} samples (from allowlist)", flush=True)
        return ds

    return sample_dataset(ds, fraction=fraction)


def apply_latency_patch_to_vllm_generate(model):
    """Replaces sync vLLM generate with engine-step-loop for accurate TTFT/TTPT.

    Uses model.llm.llm_engine.add_request() + step() loop to track:
    - TTFT: time from add_request to first step() that yields output tokens
    - TTPT: (last_output_time - first_output_time) / (num_output_tokens - 1)
    - total_latency: wall-clock from add_request to final output

    Falls back to sync generate() if engine stepping fails.
    """
    def timed_generate_inner_vllm(message, dataset=None):
        from vllm import SamplingParams
        sampling_params = model._build_vllm_sampling_params()
        req = model._build_vllm_request(message, dataset=dataset)

        # --- Engine step-loop approach for accurate TTFT ---
        try:
            engine = model.llm.llm_engine
            req_id = f"latency-{time.monotonic_ns()}"

            t_submit = time.perf_counter()
            engine.add_request(req_id, req, sampling_params)

            first_output_ts = None
            last_output_ts = None
            prev_token_count = 0
            total_output_tokens = 0
            final_text = ""
            prompt_tokens = 0
            ttft_from_metrics = True  # measured by engine step-loop

            while engine.has_unfinished_requests():
                step_outputs = engine.step()
                now = time.perf_counter()

                for o in step_outputs:
                    if hasattr(o, "prompt_token_ids") and o.prompt_token_ids:
                        prompt_tokens = max(prompt_tokens, len(o.prompt_token_ids))
                    if hasattr(o, "outputs") and o.outputs:
                        for comp in o.outputs:
                            if hasattr(comp, "token_ids") and comp.token_ids:
                                cur_count = len(comp.token_ids)
                                if cur_count > prev_token_count and prev_token_count == 0:
                                    first_output_ts = now  # first token arrived
                                prev_token_count = cur_count
                                last_output_ts = now
                            if hasattr(comp, "text") and comp.text:
                                final_text = comp.text  # cumulative, take last

            total_output_tokens = prev_token_count  # final cumulative count
            t_end = time.perf_counter()
            total_wall = t_end - t_submit

            # Compute metrics from measured timestamps
            ttft = (first_output_ts - t_submit) if first_output_ts else (total_wall * 0.3)
            if total_output_tokens > 1 and first_output_ts and last_output_ts:
                ttpt = (last_output_ts - first_output_ts) / (total_output_tokens - 1)
            else:
                ttpt = total_wall / max(total_output_tokens, 1)

        except Exception as e:
            # Fallback: use sync generate()
            print(f"[latency_test][WARN] engine-step failed ({e}), falling back to sync", flush=True)
            t_submit = time.perf_counter()
            outputs = model.llm.generate(req, sampling_params=sampling_params)
            t_end = time.perf_counter()
            total_wall = t_end - t_submit

            ttft_from_metrics = False
            first_output_ts = None
            total_output_tokens = 0
            final_text = ""
            prompt_tokens = 0
            for o in outputs:
                if hasattr(o, "prompt_token_ids") and o.prompt_token_ids:
                    prompt_tokens = len(o.prompt_token_ids)
                if hasattr(o, "outputs") and o.outputs:
                    o0 = o.outputs[0]
                    if hasattr(o0, "token_ids") and o0.token_ids:
                        total_output_tokens = len(o0.token_ids)
                    final_text = o0.text if hasattr(o0, "text") else ""
            if total_output_tokens <= 20:
                ttft = total_wall * 0.80
            elif total_output_tokens <= 100:
                ttft = total_wall * 0.40
            else:
                ttft = total_wall * 0.15
            ttpt = (total_wall - ttft) / max(total_output_tokens - 1, 1)

        model._last_ttft = ttft
        model._last_ttpt = ttpt
        model._last_num_prompt_tokens = prompt_tokens
        model._last_num_completion_tokens = total_output_tokens
        model._last_ttft_from_metrics = ttft_from_metrics

        return model._finalize_vllm_generated_text(final_text, dataset=dataset)

    model.generate_inner_vllm = timed_generate_inner_vllm
    model._last_ttft = 0.0
    model._last_ttpt = 0.0
    model._last_num_prompt_tokens = 0
    model._last_num_completion_tokens = 0
    model._last_ttft_from_metrics = False

    # --- Also patch _generate_inner_vllm for models like MiniCPM that use
    #     self.llm.chat() instead of self.llm.generate().  Since chat() is a
    #     high-level API we cannot use the engine step-loop; measure wall-clock
    #     and estimate TTFT/TTPT from total time and output length. ---
    if hasattr(model, '_generate_inner_vllm') and not hasattr(model, '_latency_patched'):
        _original_generate_inner_vllm = model._generate_inner_vllm

        def timed_generate_inner_vllm_minicpm(message, dataset=None):
            t_start = time.perf_counter()
            result = _original_generate_inner_vllm(message, dataset=dataset)
            t_end = time.perf_counter()
            total_wall = t_end - t_start

            # Extract output text for token estimation
            if isinstance(result, dict):
                output_text = result.get("prediction", result.get("detailed_prediction", ""))
            else:
                output_text = str(result)

            # Rough token count from word count
            num_tokens = max(1, len(output_text.split()) * 2)
            model._last_num_completion_tokens = num_tokens
            model._last_num_prompt_tokens = max(1, num_tokens // 2)

            # Estimate TTFT fraction from output length
            if num_tokens <= 20:
                ttft = total_wall * 0.80
            elif num_tokens <= 100:
                ttft = total_wall * 0.40
            else:
                ttft = total_wall * 0.15
            ttpt = (total_wall - ttft) / max(num_tokens - 1, 1)

            model._last_ttft = ttft
            model._last_ttpt = ttpt
            model._last_ttft_from_metrics = False  # estimated
            return result

        model._generate_inner_vllm = timed_generate_inner_vllm_minicpm
        model._latency_patched = True

    return model


def _patch_transformers_image_processor():
    """Fix transformers 4.57.6 bug: local dir paths bypass is_local flag in get_image_processor_dict.

    When pretrained_model_name_or_path is a local directory, the function sets
    is_local=True and image_processor_file to the correct local path, but then
    the if/elif/else chain falls through to the else branch which resets
    image_processor_file and calls cached_file() to download from HF Hub.
    This fails offline with an empty resolved list → IndexError.

    Also handles the case where vLLM internally resolves the model name to
    'default' (e.g., when _name_or_path is missing from config.json).
    We redirect 'default' to MODEL_PATH_QWEN25_32B env var.
    """
    import transformers.image_processing_base as ipb

    _original = ipb.ImageProcessingMixin.get_image_processor_dict

    @classmethod
    def _patched_get_image_processor_dict(
        cls,
        pretrained_model_name_or_path,
        image_processor_filename=ipb.IMAGE_PROCESSOR_NAME,
        **kwargs,
    ):
        model_path = pretrained_model_name_or_path
        # If vLLM resolves the model to 'default', redirect to actual path
        if isinstance(model_path, str) and model_path == "default":
            from pathlib import Path as _Path
            alt = os.environ.get("MODEL_PATH_QWEN25_32B", "")
            if alt and _Path(alt).is_dir():
                model_path = alt

        if os.path.isdir(model_path):
            local_file = os.path.join(model_path, image_processor_filename)
            if os.path.isfile(local_file):
                import json as _json
                with open(local_file, encoding="utf-8") as reader:
                    text = reader.read()
                image_processor_dict = _json.loads(text)
                image_processor_dict = image_processor_dict.get("image_processor", image_processor_dict)
                return image_processor_dict, kwargs
        return _original.__func__(cls, model_path, image_processor_filename=image_processor_filename, **kwargs)

    ipb.ImageProcessingMixin.get_image_processor_dict = _patched_get_image_processor_dict


def run_latency_tests(config: dict[str, Any]):
    """Main latency test runner."""
    # --- Setup paths ---
    repo_root = str(config.get("repo_root", os.getcwd())).replace("{repo_root}", os.getcwd())
    if not os.path.isdir(repo_root):
        repo_root = os.getcwd()
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # --- Environment setup ---
    # Route to correct minimal config based on model framework.
    # Qwen models use VLMEVAL_USE_QWEN_MINIMAL_CONFIG (default).
    # MiniCPM uses VLMEVAL_USE_MINICPM45_MINIMAL_CONFIG.
    # Gemma3 uses VLMEVAL_USE_GEMMA3_MINIMAL_CONFIG.
    model_framework = config.get("model_framework", "qwen").strip().lower()
    if model_framework == "minicpm":
        os.environ.setdefault("VLMEVAL_USE_MINICPM45_MINIMAL_CONFIG", "1")
        os.environ.pop("VLMEVAL_USE_QWEN_MINIMAL_CONFIG", None)
        # MiniCPM reads max_new_tokens from MINICPM45_MAX_NEW_TOKENS env var (default 16384)
        # Align with smoke test's models.yaml: MINICPM45_MAX_NEW_TOKENS='16384'
        os.environ.setdefault("MINICPM45_MAX_NEW_TOKENS", "16384")
    elif model_framework == "gemma3":
        os.environ.setdefault("VLMEVAL_USE_GEMMA3_MINIMAL_CONFIG", "1")
        os.environ.pop("VLMEVAL_USE_QWEN_MINIMAL_CONFIG", None)
        # Gemma3 config_gemma3_minimal.py reads GEMMA3_MAX_NEW_TOKENS at import time (default 4096)
        # Set before import to align with smoke test
        os.environ.setdefault("GEMMA3_MAX_NEW_TOKENS", "4096")
    else:
        os.environ.setdefault("VLMEVAL_USE_QWEN_MINIMAL_CONFIG", "1")
    os.environ.setdefault("VLMEVAL_VLM_MINIMAL_IMPORT", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # Avoid HuggingFace download attempts when model files exist locally.
    # transformers 4.57.6 has a bug where local dir paths still trigger
    # cached_file() download in get_image_processor_dict (IndexError on empty list).
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # Monkey-patch transformers get_image_processor_dict to fix local dir handling.
    # In transformers 4.57.6, when pretrained_model_name_or_path is a local dir,
    # is_local is set but the else branch resets image_processor_file and tries
    # to download via cached_file(), which fails offline.
    _patch_transformers_image_processor()

    # Multi-GPU setup
    gpu_ids = config.get("node_gpu_ids", "0")
    tp_size = config.get("tensor_parallel_size", None)
    if tp_size is None:
        tp_size = len(gpu_ids.split(",")) if "," in gpu_ids else 1
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    os.environ["VLLM_TP_SIZE"] = str(tp_size)
    print(f"[latency_test] GPU: {gpu_ids} (CUDA_VISIBLE_DEVICES), TP size: {tp_size}", flush=True)

    model_path = config.get("model_path", "/models/Qwen2.5-VL-32B-Instruct")
    os.environ["MODEL_PATH"] = model_path
    os.environ["MODEL_ROOT"] = os.path.dirname(model_path)
    # Set all Qwen model path env vars so any Qwen registry can find the model
    os.environ["MODEL_PATH_QWEN25_3B"] = model_path
    os.environ["MODEL_PATH_QWEN25"] = model_path
    os.environ["MODEL_PATH_QWEN25_32B"] = model_path
    os.environ["MODEL_PATH_QWEN25_72B"] = model_path

    # --- Parameters ---
    datasets_list = config["datasets"]
    replay_modes = config["replay_modes"]
    cot_modes = config["cot_modes"]
    sampling_fraction = config.get("sampling_fraction", 0.01)
    results_root = config.get("results_root", "runs/latency_test_qwen25vl32b")
    model_name = config.get("model_name", "Qwen2.5-VL-32B-Instruct-Replay")
    batch_size = config.get("batch_size", 1)

    replay_cfg = config.get("replay", {})
    replay_times = replay_cfg.get("replay_times", 1)
    template_on_last = replay_cfg.get("template_on_last_replay_text", 1)
    image_copy_mode = replay_cfg.get("image_copy_mode", "reuse_path")
    limit_mm = replay_cfg.get("limit_mm_per_prompt", 2)
    safe_fallback = replay_cfg.get("safe_fallback", 1)
    safe_truncate_chars = replay_cfg.get("safe_truncate_chars", 6000)

    os.makedirs(results_root, exist_ok=True)

    # --- Import vlmeval ---
    # Clear cached config modules so framework switch (qwen→minicpm→gemma3) works
    import sys as _sys
    for _mod in list(_sys.modules):
        if 'vlmeval.config' in _mod:
            del _sys.modules[_mod]
    from vlmeval.config_runtime import supported_VLM

    # --- Import vlmeval ---
    # Clear cached config modules so framework switch (qwen→minicpm→gemma3) works
    for _mod in list(_sys.modules):
        if 'vlmeval.config' in _mod:
            del _sys.modules[_mod]
    from vlmeval.config_runtime import supported_VLM
    from vlmeval.dataset import build_dataset
    from vlmeval.inference import _maybe_build_prompt_struct, _build_replay_meta, _attach_replay_meta

    # --- Build model once ---
    print(f"[latency_test] Loading model: {model_name} from {model_path}", flush=True)
    print(f"[latency_test] GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}, TP size: {tp_size}", flush=True)

    # Build model with kwargs matching run_benchmark's env-var approach.
    # run_benchmark sets VLLM_TP_SIZE, VLLM_MAX_NUM_SEQS, VLLM_MAX_MODEL_LEN,
    # VLLM_GPU_MEMORY_UTILIZATION as env vars; the model picks them up.
    # We mirror this by both setting env vars and passing explicit kwargs.
    #
    # max_new_tokens per framework (aligned with smoke test / models.yaml):
    #   Qwen:    2048 (code default, passed via kwarg)
    #   MiniCPM: 16384 (MINICPM45_MAX_NEW_TOKENS env var, set above)
    #   Gemma3:  4096 (GEMMA3_MAX_NEW_TOKENS env var, set above; partial binds it)
    max_new_tokens = config.get("max_new_tokens", 2048)
    model_kwargs: dict[str, Any] = {
        "tensor_parallel_size": tp_size,
        "model_path": model_path,  # Explicit override: avoids module-cached path from previous model
    }
    # Qwen uses kwarg; MiniCPM ignores kwarg (uses env); Gemma3 partial already binds it
    if model_framework != "gemma3":
        model_kwargs["max_new_tokens"] = max_new_tokens
    vllm_cfg = config.get("vllm", {})
    if vllm_cfg.get("max_num_seqs"):
        os.environ["VLLM_MAX_NUM_SEQS"] = str(vllm_cfg["max_num_seqs"])
    if vllm_cfg.get("max_model_len"):
        model_kwargs["max_model_len"] = vllm_cfg["max_model_len"]
    if vllm_cfg.get("gpu_memory_utilization"):
        model_kwargs["gpu_utils"] = vllm_cfg["gpu_memory_utilization"]

    print(f"[latency_test] Model kwargs: { {k: v for k, v in model_kwargs.items() if k != 'model_path'} }", flush=True)
    model = supported_VLM[model_name](**model_kwargs)
    print(f"[latency_test] Model loaded with TP={tp_size}, max_new_tokens={max_new_tokens}.", flush=True)

    # Apply timing patch to vLLM generate method
    model = apply_latency_patch_to_vllm_generate(model)

    # --- Results storage ---
    all_results: list[dict] = []
    all_inputs: list[dict] = []
    all_outputs: list[dict] = []
    all_prompts: list[dict] = []  # Dump prompts for inspection

    for replay_mode in replay_modes:
        for cot_mode in cot_modes:
            config_name = f"{replay_mode}_{cot_mode}"
            print(f"\n{'='*60}")
            print(f"[latency_test] Config: {config_name}")
            print(f"{'='*60}", flush=True)

            # --- Set environment for this config ---
            os.environ["REPLAY_MODE"] = replay_mode
            os.environ["REPLAY_TIMES"] = str(replay_times)
            os.environ["REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT"] = str(template_on_last)
            os.environ["REPLAY_IMAGE_COPY_MODE"] = image_copy_mode
            os.environ["REPLAY_LIMIT_MM_PER_PROMPT"] = str(limit_mm)
            os.environ["REPLAY_SAFE_FALLBACK"] = str(safe_fallback)
            os.environ["REPLAY_SAFE_TRUNCATE_CHARS"] = str(safe_truncate_chars)

            if cot_mode == "noCoT":
                os.environ["REPLAY_PROMPT_TEMPLATE_NAME"] = "directly_answer"
            else:
                os.environ["REPLAY_PROMPT_TEMPLATE_NAME"] = "identity"

            # Dataset-specific prompt schemas (aligned with run_benchmark)
            # DynaMath: use legacy_two_keys for Qwen2.5-VL to match official table
            if model_framework == "qwen":
                os.environ["DYNAMATH_PROMPT_SCHEMA"] = "legacy_two_keys"

            # Re-read replay config from updated env
            from vlmeval.vlm.replay_policy import read_replay_config_from_env
            model.replay_cfg = read_replay_config_from_env()

            if model_framework == "qwen":
                from vlmeval.vlm.qwen2_vl.replay_prompt_template import read_prompt_template_config_from_env
                model.prompt_template_cfg = read_prompt_template_config_from_env()

            # --- Clear KV cache AND MM (image encoder) cache between configs ---
            if hasattr(model, 'llm'):
                if hasattr(model.llm, 'reset_prefix_cache'):
                    model.llm.reset_prefix_cache()
                if hasattr(model.llm, 'reset_mm_cache'):
                    model.llm.reset_mm_cache()
                print(f"[latency_test] KV + MM cache cleared for {config_name}", flush=True)

            # --- Warmup: run first sample with fresh cache, discard result ---
            warmup_done = False

            # Load dataset index allowlists if specified (aligns with smoke_test)
            dataset_allowlists: dict[str, list[int]] = {}
            raw_allowlists = config.get("dataset_index_allowlists", {})
            if raw_allowlists:
                import json as _json
                for ds_name, path in raw_allowlists.items():
                    resolved = str(path).replace("{repo_root}", repo_root)
                    if os.path.isfile(resolved):
                        with open(resolved) as f:
                            dataset_allowlists[ds_name] = _json.load(f)
                        print(f"[latency_test] Loaded allowlist for {ds_name}: {len(dataset_allowlists[ds_name])} samples", flush=True)

            # --- Run each dataset ---
            for dataset_name in datasets_list:
                print(f"\n[latency_test] Dataset: {dataset_name}", flush=True)
                try:
                    allowlist = dataset_allowlists.get(dataset_name)
                    ds = build_latency_dataset(dataset_name, fraction=sampling_fraction, allowlist=allowlist)
                except Exception as e:
                    print(f"[ERROR] Failed to build dataset {dataset_name}: {e}", flush=True)
                    continue

                ds_name = ds.dataset_name
                data = ds.data
                n_samples = len(data)

                # Build index→position map and prompt_cache for replay_meta (aligned with run_benchmark)
                index_to_position = {
                    str(data.iloc[pos].get("index", pos)): pos
                    for pos in range(len(data))
                }
                prompt_cache: dict[int, list[dict[str, Any]]] = {}

                ds_latencies: list[float] = []
                ds_ttfts: list[float] = []
                ds_ttpts: list[float] = []
                ds_e2e_latencies: list[float] = []

                for i in range(n_samples):
                    row = data.iloc[i]
                    idx = str(row.get("index", i))

                    try:
                        struct = _maybe_build_prompt_struct(model, ds, ds_name, row)
                    except Exception:
                        struct = ds.build_prompt(row)

                    # Attach replay_meta (aligned with run_benchmark's infer_data)
                    replay_meta = _build_replay_meta(
                        model=model,
                        dataset=ds,
                        dataset_name=ds_name,
                        current_row=row,
                        current_struct=struct,
                        index_to_position=index_to_position,
                        prompt_cache=prompt_cache,
                    )
                    struct = _attach_replay_meta(struct, replay_meta)

                    # Record prompt for inspection (first 2 per config/dataset)
                    if len(all_prompts) < 2 or True:  # Always record
                        all_prompts.append({
                            "config": config_name,
                            "replay_mode": replay_mode,
                            "cot_mode": cot_mode,
                            "dataset": ds_name,
                            "sample_index": idx,
                            "question": str(row.get("question", ""))[:3000],
                            "struct": struct,
                            "replay_meta": replay_meta,
                        })

                    try:
                        t_generate_start = time.perf_counter()
                        response = model.generate(message=struct, dataset=ds_name)
                        t_generate_end = time.perf_counter()
                        e2e_latency = t_generate_end - t_generate_start
                    except Exception as e:
                        print(f"[ERROR] sample {idx}: {e}", flush=True)
                        response = f"[FAILED] {e}"
                        e2e_latency = 0.0

                    # Retrieve accurate TTFT/TTPT from engine step-loop (set by timed_generate_inner_vllm)
                    ttft = getattr(model, "_last_ttft", 0.0)
                    ttpt = getattr(model, "_last_ttpt", 0.0)
                    num_prompt_tokens = getattr(model, "_last_num_prompt_tokens", 0)
                    num_completion_tokens = getattr(model, "_last_num_completion_tokens", 0)

                    if isinstance(response, str):
                        output_text = response
                    elif isinstance(response, dict):
                        output_text = response.get("prediction", str(response))
                    else:
                        output_text = str(response)

                    # Use vLLM token counts if available, otherwise estimate
                    if num_completion_tokens <= 0:
                        num_completion_tokens = max(1, len(output_text.split()) * 2)
                    if num_prompt_tokens <= 0:
                        num_prompt_tokens = max(1, num_completion_tokens)

                    # Compute latency from TTFT + TTPT * (output_tokens - 1)
                    # This is more reproducible than wall-clock as it excludes
                    # Python-level preprocessing/postprocessing overhead.
                    total_latency = ttft + ttpt * max(num_completion_tokens - 1, 0)

                    # Fallback: if patch-based metrics are zero (e.g. unsupported
                    # model path), use end-to-end wall-clock as total latency.
                    if total_latency <= 0 and e2e_latency > 0:
                        total_latency = e2e_latency
                        ttft = e2e_latency * 0.15
                        ttpt = (e2e_latency - ttft) / max(num_completion_tokens - 1, 1)

                    # Skip first sample of each config as warmup (cold cache)
                    if not warmup_done:
                        warmup_done = True
                        print(f"  [warmup/{n_samples}] e2e={e2e_latency:.3f}s, "
                              f"latency={total_latency:.3f}s (discarded)",
                              flush=True)
                        # Warmup: save input/output but skip latency metrics
                        all_inputs.append({
                            "config": config_name,
                            "dataset": ds_name,
                            "sample_index": idx,
                            "warmup": True,
                            "question": str(row.get("question", ""))[:2000],
                            "image_path": str(row.get("image_path", row.get("image", ""))),
                            "prompt_struct": str(struct)[:3000],
                        })
                        all_outputs.append({
                            "config": config_name,
                            "dataset": ds_name,
                            "sample_index": idx,
                            "warmup": True,
                            "output": output_text,
                        })
                        continue

                    record = {
                        "config": config_name,
                        "replay_mode": replay_mode,
                        "cot_mode": cot_mode,
                        "dataset": ds_name,
                        "sample_index": idx,
                        "e2e_latency_s": round(e2e_latency, 4),
                        "total_latency_s": round(total_latency, 4),
                        "ttft_s": round(ttft, 4),
                        "ttpt_s": round(ttpt, 6),
                        "num_output_tokens": int(num_completion_tokens),
                        "num_input_tokens": int(num_prompt_tokens),
                        "input_preview": str(struct)[:500],
                        "output_preview": output_text[:2000],
                    }
                    all_results.append(record)
                    ds_latencies.append(total_latency)
                    ds_ttfts.append(ttft)
                    ds_ttpts.append(ttpt)
                    ds_e2e_latencies.append(e2e_latency)

                    all_inputs.append({
                        "config": config_name,
                        "dataset": ds_name,
                        "sample_index": idx,
                        "question": str(row.get("question", ""))[:2000],
                        "image_path": str(row.get("image_path", row.get("image", ""))),
                        "prompt_struct": str(struct)[:3000],
                    })
                    all_outputs.append({
                        "config": config_name,
                        "dataset": ds_name,
                        "sample_index": idx,
                        "output": output_text,
                    })

                    ttft_src = "M" if getattr(model, "_last_ttft_from_metrics", False) else "E"
                    print(f"  [{i+1}/{n_samples}] e2e={e2e_latency:.3f}s, "
                          f"latency={total_latency:.3f}s, "
                          f"ttft={ttft:.3f}s[{ttft_src}], ttpt={ttpt:.4f}s, "
                          f"tokens_out={num_completion_tokens}", flush=True)

                print(f"  [{config_name}] {ds_name}: "
                      f"avg_e2e={np.mean(ds_e2e_latencies):.3f}s±{np.std(ds_e2e_latencies):.3f}s, "
                      f"avg_latency={np.mean(ds_latencies):.3f}s±{np.std(ds_latencies):.3f}s, "
                      f"avg_ttft={np.mean(ds_ttfts):.3f}s, "
                      f"avg_ttpt={np.mean(ds_ttpts):.5f}s",
                      flush=True)

    # ============================================================
    # Save all results
    # ============================================================
    results_df = pd.DataFrame(all_results)
    results_csv = os.path.join(results_root, "all_latency_results.csv")
    results_df.to_csv(results_csv, index=False)

    inputs_df = pd.DataFrame(all_inputs)
    inputs_csv = os.path.join(results_root, "all_inputs.csv")
    inputs_df.to_csv(inputs_csv, index=False)

    outputs_df = pd.DataFrame(all_outputs)
    outputs_csv = os.path.join(results_root, "all_outputs.csv")
    outputs_df.to_csv(outputs_csv, index=False)

    # Save prompts for inspection
    import json as _json
    prompts_json = os.path.join(results_root, "all_prompts.json")
    with open(prompts_json, "w", encoding="utf-8") as f:
        _json.dump(all_prompts, f, ensure_ascii=False, indent=2)
    # Also save a compact view: just the text prompts per config
    prompts_txt = os.path.join(results_root, "all_prompts.txt")
    with open(prompts_txt, "w", encoding="utf-8") as f:
        for p in all_prompts:
            f.write(f"{'='*80}\n")
            f.write(f"Config: {p['config']} | Dataset: {p['dataset']} | Sample: {p['sample_index']}\n")
            f.write(f"Replay Meta: {_json.dumps(p['replay_meta'], ensure_ascii=False)}\n")
            f.write(f"--- Struct ---\n")
            for item in p['struct']:
                t = item.get('type', '?')
                if t == 'text':
                    f.write(f"[text] {item.get('value', item.get('text', ''))}\n")
                elif t == 'image':
                    val = str(item.get('value', item.get('image', '')))
                    f.write(f"[image] {val[:200]}{'...(truncated)' if len(val) > 200 else ''}\n")
                else:
                    f.write(f"[{t}] {_json.dumps(item, ensure_ascii=False)[:500]}\n")
            f.write(f"--- Original Question ---\n{p.get('question', 'N/A')}\n")
            f.write("\n")

    print(f"\n[latency_test] Results saved to {results_root}/")
    print(f"  - all_latency_results.csv ({len(results_df)} rows)")
    print(f"  - all_inputs.csv ({len(inputs_df)} rows)")
    print(f"  - all_outputs.csv ({len(outputs_df)} rows)")
    print(f"  - all_prompts.json ({len(all_prompts)} prompts for inspection)")
    print(f"  - all_prompts.txt (human-readable prompt dump)")

    # ============================================================
    # Per-dataset per-config averages
    # ============================================================
    print("\n" + "=" * 90)
    print("PER-DATASET PER-CONFIG AVERAGES")
    print("=" * 90)

    per_ds_rows = []
    for (cfg, ds_name), grp in results_df.groupby(["config", "dataset"]):
        row = {
            "config": cfg,
            "replay_mode": grp["replay_mode"].iloc[0],
            "cot_mode": grp["cot_mode"].iloc[0],
            "dataset": ds_name,
            "n_samples": len(grp),
            "avg_e2e_latency_s": round(grp["e2e_latency_s"].mean(), 4),
            "std_e2e_latency_s": round(grp["e2e_latency_s"].std(), 4),
            "avg_total_latency_s": round(grp["total_latency_s"].mean(), 4),
            "std_total_latency_s": round(grp["total_latency_s"].std(), 4),
            "avg_ttft_s": round(grp["ttft_s"].mean(), 4),
            "std_ttft_s": round(grp["ttft_s"].std(), 4),
            "avg_ttpt_s": round(grp["ttpt_s"].mean(), 6),
            "std_ttpt_s": round(grp["ttpt_s"].std(), 6),
            "avg_num_output_tokens": round(grp["num_output_tokens"].mean(), 1),
        }
        per_ds_rows.append(row)

    per_ds_df = pd.DataFrame(per_ds_rows)
    per_ds_csv = os.path.join(results_root, "per_dataset_summary.csv")
    per_ds_df.to_csv(per_ds_csv, index=False)

    print(f"{'Config':<30} {'Dataset':<26} {'E2E(s)':>8} {'±Std':>7} {'Lat(s)':>8} {'TTPT(s)':>9} {'#Tok':>5} {'N':>4}")
    print("-" * 105)
    for row in per_ds_rows:
        print(f"{row['config']:<30} {row['dataset']:<26} {row['avg_e2e_latency_s']:>8.3f} "
              f"{row['std_e2e_latency_s']:>7.3f} {row['avg_total_latency_s']:>8.3f} "
              f"{row['avg_ttpt_s']:>9.5f} "
              f"{row['avg_num_output_tokens']:>5.0f} {row['n_samples']:>4}")

    # ============================================================
    # Overall (sample-level) averages
    # ============================================================
    print("\n" + "=" * 90)
    print("OVERALL SAMPLE-LEVEL AVERAGES")
    print("=" * 90)

    overall_rows = []
    for cfg, grp in results_df.groupby("config"):
        row = {
            "config": cfg,
            "replay_mode": grp["replay_mode"].iloc[0],
            "cot_mode": grp["cot_mode"].iloc[0],
            "n_samples": len(grp),
            "n_datasets": grp["dataset"].nunique(),
            "avg_e2e_latency_s": round(grp["e2e_latency_s"].mean(), 4),
            "std_e2e_latency_s": round(grp["e2e_latency_s"].std(), 4),
            "avg_total_latency_s": round(grp["total_latency_s"].mean(), 4),
            "std_total_latency_s": round(grp["total_latency_s"].std(), 4),
            "avg_ttft_s": round(grp["ttft_s"].mean(), 4),
            "std_ttft_s": round(grp["ttft_s"].std(), 4),
            "avg_ttpt_s": round(grp["ttpt_s"].mean(), 6),
            "std_ttpt_s": round(grp["ttpt_s"].std(), 6),
            "avg_num_output_tokens": round(grp["num_output_tokens"].mean(), 1),
        }
        overall_rows.append(row)

    overall_df = pd.DataFrame(overall_rows)
    overall_csv = os.path.join(results_root, "overall_summary.csv")
    overall_df.to_csv(overall_csv, index=False)

    print(f"{'Config':<30} {'E2E(s)':>8} {'±Std':>7} {'Lat(s)':>8} {'TTPT(s)':>9} {'#Tok':>5} {'Samp':>5} {'DS':>3}")
    print("-" * 80)
    for row in overall_rows:
        print(f"{row['config']:<30} {row['avg_e2e_latency_s']:>8.3f} {row['std_e2e_latency_s']:>7.3f} "
              f"{row['avg_total_latency_s']:>8.3f} "
              f"{row['avg_ttpt_s']:>9.5f} "
              f"{row['avg_num_output_tokens']:>5.0f} {row['n_samples']:>5} {row['n_datasets']:>3}")

    # ============================================================
    # Speedup analysis
    # ============================================================
    print("\n" + "=" * 90)
    print("SPEEDUP ANALYSIS (vs baseline: image_text_noCoT)")
    print("=" * 90)

    baseline = None
    for row in overall_rows:
        if row["replay_mode"] == "image_text" and row["cot_mode"] == "noCoT":
            baseline = row
            break

    if baseline:
        bl = baseline["avg_e2e_latency_s"]
        bl_ttpt = baseline["avg_ttpt_s"]
        print(f"Baseline: {baseline['config']} → e2e={bl:.3f}s, ttpt={bl_ttpt:.5f}s")
        print(f"{'Config':<30} {'E2E(s)':>8} {'Ratio':>8} {'TTPT(s)':>10} {'TTPT Ratio':>10}")
        print("-" * 68)
        for row in overall_rows:
            ratio = row["avg_e2e_latency_s"] / max(bl, 0.001)
            ttpt_ratio = row["avg_ttpt_s"] / max(bl_ttpt, 0.000001)
            print(f"{row['config']:<30} {row['avg_e2e_latency_s']:>8.3f} {ratio:>8.2f}x "
                  f"{row['avg_ttpt_s']:>10.5f} {ttpt_ratio:>10.2f}x")

    # Explicitly release vLLM engine and GPU memory
    if hasattr(model, 'llm') and model.llm is not None:
        try:
            del model.llm
        except Exception:
            pass
    del model
    import gc; gc.collect()
    try:
        import torch; torch.cuda.empty_cache(); torch.cuda.synchronize()
    except Exception:
        pass
    print("[latency_test] Model unloaded, GPU memory released.", flush=True)

    return results_df, per_ds_df, overall_df


def dump_prompts_only(config: dict[str, Any]):
    """Dump prompts for all config/dataset combinations without loading model."""
    import json as _json

    os.environ.setdefault("VLMEVAL_USE_QWEN_MINIMAL_CONFIG", "1")
    os.environ.setdefault("VLMEVAL_VLM_MINIMAL_IMPORT", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from vlmeval.dataset import build_dataset

    datasets_list = config["datasets"]
    replay_modes = config["replay_modes"]
    cot_modes = config["cot_modes"]
    sampling_fraction = config.get("sampling_fraction", 0.01)
    results_root = config.get("results_root", "runs/latency_test_qwen25vl32b")
    replay_cfg = config.get("replay", {})
    os.makedirs(results_root, exist_ok=True)

    all_prompts: list[dict] = []

    for replay_mode in replay_modes:
        for cot_mode in cot_modes:
            config_name = f"{replay_mode}_{cot_mode}"
            template_name = "directly_answer" if cot_mode == "noCoT" else "identity"

            for dataset_name in datasets_list:
                print(f"[dump] {config_name} | {dataset_name}", flush=True)
                try:
                    ds = build_dataset(dataset_name)
                    ds = sample_dataset(ds, fraction=sampling_fraction)
                except Exception as e:
                    print(f"  [SKIP] {e}", flush=True)
                    continue

                data = ds.data
                ds_name = ds.dataset_name
                n_show = min(3, len(data))

                for i in range(n_show):
                    row = data.iloc[i]
                    idx = str(row.get("index", i))
                    try:
                        struct = ds.build_prompt(row)
                    except Exception:
                        struct = [{"type": "text", "value": str(row.get("question", ""))}]

                    prompt_info = {
                        "config": config_name,
                        "replay_mode": replay_mode,
                        "cot_mode": cot_mode,
                        "prompt_template": template_name,
                        "dataset": ds_name,
                        "sample_index": idx,
                        "raw_question": str(row.get("question", ""))[:3000],
                        "raw_answer": str(row.get("answer", ""))[:500],
                        "hint": str(row.get("hint", ""))[:500] if "hint" in row else "",
                        "struct": struct,
                    }
                    all_prompts.append(prompt_info)

    # Save JSON
    prompts_json = os.path.join(results_root, "all_prompts_dryrun.json")
    with open(prompts_json, "w", encoding="utf-8") as f:
        _json.dump(all_prompts, f, ensure_ascii=False, indent=2)

    # Save human-readable TXT
    prompts_txt = os.path.join(results_root, "all_prompts_dryrun.txt")
    with open(prompts_txt, "w", encoding="utf-8") as f:
        for p in all_prompts:
            f.write(f"{'='*80}\n")
            f.write(f"Config: {p['config']} (template={p['prompt_template']}) | "
                    f"Dataset: {p['dataset']} | Sample: {p['sample_index']}\n")
            f.write(f"Answer: {p['raw_answer']}\n")
            if p['hint']:
                f.write(f"Hint: {p['hint']}\n")
            f.write(f"--- Replay Mode ---\n")
            f.write(f"  replay_mode={p['replay_mode']}\n")
            f.write(f"  replay_times={replay_cfg.get('replay_times',1)} template_on_last={replay_cfg.get('template_on_last_replay_text',1)}\n")
            f.write(f"--- Prompt Struct (dataset.build_prompt, before replay processing) ---\n")
            for item in p['struct']:
                t = item.get('type', '?')
                if t == 'text':
                    f.write(f"[text]\n{item.get('value', item.get('text', ''))}\n\n")
                elif t == 'image':
                    val = str(item.get('value', item.get('image', '')))
                    f.write(f"[image] {val[:200]}{'...(truncated)' if len(val) > 200 else ''}\n\n")
                else:
                    f.write(f"[{t}] {_json.dumps(item, ensure_ascii=False)[:500]}\n\n")
            f.write(f"--- Template that will be applied during inference ---\n")
            f.write(f"  CoT mode: {p['cot_mode']} → prompt_template_name: {p['prompt_template']}\n")
            if p['cot_mode'] == 'noCoT':
                f.write(f"  Template text: '{{problem}}\\nAnswer directly with a single word or short phrase.\\nDo not output any explanation, derivation, words, or extra symbols.'\n")
            else:
                f.write(f"  Template text: '{{problem}}' (identity — no modification)\n")
            f.write("\n")

    print(f"\n[dump] Saved {len(all_prompts)} prompts:")
    print(f"  JSON: {prompts_json}")
    print(f"  TXT:  {prompts_txt}")
    print(f"\n[dump] NOTE: These are dataset raw prompts BEFORE replay processing.")
    print(f"[dump] The actual prompt will be modified by:")
    print(f"  - Replay pipeline: duplicate images/text according to replay_mode ({', '.join(replay_modes)})")
    print(f"  - Prompt template: 'directly_answer' adds short-answer instruction; 'identity' keeps unchanged")
    print(f"[dump] To see the final prompt, run the full latency test and check all_prompts.txt")


def main():
    parser = argparse.ArgumentParser(description="VLMEvalKit Latency Test")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Override model path")
    parser.add_argument("--fraction", type=float, default=None,
                        help="Override sampling fraction (e.g., 0.01 for 1%%)")
    parser.add_argument("--gpu", type=str, default=None,
                        help="Override GPU IDs (e.g., 0 or 0,1,2,3)")
    parser.add_argument("--tp-size", type=int, default=None,
                        help="Override tensor parallel size")
    parser.add_argument("--datasets", type=str, nargs="*", default=None,
                        help="Override datasets to test")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Override model registry name")
    parser.add_argument("--model-framework", type=str, default=None,
                        help="Model framework: qwen, minicpm, gemma3")
    parser.add_argument("--dump-prompts-only", action="store_true",
                        help="Dump prompts to file without loading model (quick preview)")
    args = parser.parse_args()

    config = load_yaml_config(args.config)

    if args.model_path:
        config["model_path"] = args.model_path
    if args.model_name:
        config["model_name"] = args.model_name
    if args.model_framework:
        config["model_framework"] = args.model_framework
    if args.fraction is not None:
        config["sampling_fraction"] = args.fraction
    if args.gpu is not None:
        config["node_gpu_ids"] = args.gpu
    if args.tp_size is not None:
        config["tensor_parallel_size"] = args.tp_size
    if args.datasets:
        config["datasets"] = args.datasets

    print("=" * 60)
    print("VLMEvalKit Latency Test")
    print("=" * 60)
    print(f"Config file:     {args.config}")
    print(f"Model path:      {config.get('model_path', 'auto')}")
    print(f"GPU:             {config.get('node_gpu_ids', 'auto')}")
    print(f"Datasets:        {config['datasets']}")
    print(f"Replay modes:    {config['replay_modes']}")
    print(f"CoT modes:       {config['cot_modes']}")
    print(f"Sampling:        {config.get('sampling_fraction', 0.01)*100:.1f}%")
    print("=" * 60, flush=True)

    if "models" in config:
        # Multi-model mode: run each model in a fresh subprocess so
        # torch.cuda.device_count() is not cached from previous models.
        import subprocess, tempfile
        for i, m in enumerate(config["models"]):
            name = m["name"]
            print(f"\n{'#'*60}")
            print(f"# Model {i+1}/{len(config['models'])}: {name}")
            print(f"{'#'*60}\n", flush=True)

            single_config = dict(config)
            del single_config["models"]
            single_config["model_path"] = m["path"]
            single_config["model_name"] = m["registry"]
            single_config["model_framework"] = m["framework"]
            single_config["node_gpu_ids"] = str(m["gpu"])
            single_config["tensor_parallel_size"] = m["tp"]
            single_config["results_root"] = f"{config['results_root']}/{name}"

            import yaml as _yaml
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tf:
                _yaml.dump(single_config, tf)
                tmp_config = tf.name

            cmd = [sys.executable, __file__, "--config", tmp_config]
            if args.dump_prompts_only:
                cmd.append("--dump-prompts-only")
            env = dict(os.environ)
            env.setdefault("HF_HUB_OFFLINE", "1")
            env.setdefault("TRANSFORMERS_OFFLINE", "1")
            rc = subprocess.run(cmd, env=env).returncode
            os.unlink(tmp_config)
            if rc != 0:
                print(f"[ERROR] Model {name} failed with rc={rc}", flush=True)
    elif args.dump_prompts_only:
        dump_prompts_only(config)
    else:
        run_latency_tests(config)


if __name__ == "__main__":
    main()
