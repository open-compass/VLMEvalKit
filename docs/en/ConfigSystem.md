
# Config System

By default, VLMEvalKit starts evaluation by specifying `--model` and `--data` in `run.py`: the model name (defined in `vlmeval/config/__init__.py`) and the dataset name (defined in `vlmeval/dataset/__init__.py` or `vlmeval/dataset/VideoBench/video_dataset_config.py`). This is simple and efficient for most cases, but it becomes less flexible when you want to evaluate many models/datasets with different settings.

To address this, VLMEvalKit provides a more flexible config system. You can specify model and dataset settings in a JSON file and pass it to `run.py` via `--config`. Below is an example:

```json
{
    "model": {
        "GPT4o_20240806_T00_HIGH": {
            "class": "GPT4V",
            "model": "gpt-4o-2024-08-06",
            "temperature": 0,
            "img_detail": "high"
        },
        "GPT4o_20240806_T10_Low": {
            "class": "GPT4V",
            "model": "gpt-4o-2024-08-06",
            "temperature": 1.0,
            "img_detail": "low"
        },
        "GPT4o_20241120": {}
    },
    "data": {
        "MME-RealWorld-Lite": {
            "class": "MMERealWorld",
            "dataset": "MME-RealWorld-Lite"
        },
        "MMBench_DEV_EN_V11": {
            "class": "ImageMCQDataset",
            "dataset": "MMBench_DEV_EN_V11"
        },
        "MMBench_Video_8frame_nopack": {},
        "Video-MME_16frame_subs": {
            "class": "VideoMME",
            "dataset": "Video-MME",
            "nframe": 16,
            "use_subtitle": true
        }
    }
}
```

Explanation:

1. Two top-level fields are supported: `model` and `data`. Each field is a dict. Keys are user-defined names for models/datasets; values are the settings.
2. For an item under `model`, the value is a dict with:
    - `class`: the model class name defined in `vlmeval/vlm/__init__.py` (open-source models) or `vlmeval/api/__init__.py` (API models)
    - other kwargs: model-specific parameters; see the model class definition (e.g., `model`, `temperature`, `img_detail` for `GPT4V`). Most model classes require `model`.
    - Tip: models already defined in `supported_VLM` in `vlmeval/config/__init__.py` can be used as keys with an empty value. For example, `GPT4o_20240806_T00_HIGH: {}` is equivalent to providing the full kwargs like `{'class': 'GPT4V', 'model': 'gpt-4o-2024-08-06', 'temperature': 0, 'img_size': -1, 'img_detail': 'high', 'retry': 10, 'verbose': False}`.
3. For the `data` dict, it is recommended to use official dataset names as keys (or key parts), because post-processing and heuristics often depend on dataset names. For an item under `data`, the value is a dict with:
    - `class`: the dataset class name defined in `vlmeval/dataset/__init__.py`
    - other kwargs: dataset-specific parameters; see the dataset class definition. Most datasets require `dataset`. Most video datasets require `nframe` or `fps`.
    - Tip: datasets already defined in `supported_video_dataset` in `vlmeval/dataset/VideoBench/video_dataset_config.py` can be used as keys with an empty value. For example, `MMBench_Video_8frame_nopack: {}` is equivalent to `{'class': 'MMBenchVideo', 'dataset': 'MMBench-Video', 'nframe': 8, 'pack': False}`.

Save the example JSON as `config.json`, then run:

```bash
python run.py --config config.json
```

This will generate outputs under `$WORK_DIR` (pattern: `{$WORK_DIR}/{$MODEL_NAME}/{$MODEL_NAME}_{$DATASET_NAME}_*`), for example:

- `$WORK_DIR/GPT4o_20240806_T00_HIGH/GPT4o_20240806_T00_HIGH_MME-RealWorld-Lite*`
- `$WORK_DIR/GPT4o_20240806_T10_Low/GPT4o_20240806_T10_Low_MME-RealWorld-Lite*`
- `$WORK_DIR/GPT4o_20240806_T00_HIGH/GPT4o_20240806_T00_HIGH_MMBench_DEV_EN_V11*`
- `$WORK_DIR/GPT4o_20240806_T10_Low/GPT4o_20240806_T10_Low_MMBench_DEV_EN_V11*`
- ...
