# Config System

By default, VLMEvalKit launches the evaluation by setting the model name(s) (defined in `/vlmeval/config.py`) and dataset name(s) (defined in `vlmeval/dataset/__init__.py` or `vlmeval/dataset/video_dataset_config.py`) in the `run.py` script with the `--model` and `--data` arguments. Such approach is simple and efficient in most scenarios, however, it may not be flexible enough when the user wants to evaluate multiple models / datasets with different settings.

To address this, VLMEvalKit provides a more flexible config system. The user can specify the model and dataset settings in a json file, and pass the path to the config file to the `run.py` script with the `--config` argument. Here is a sample config json:

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
        "MMBench_Video_8frame_nopack": {
            "preset": "MMBench_Video_8frame_nopack"
        },
        "Video-MME_16frame_subs": {
            "class": "VideoMME",
            "dataset": "Video-MME",
            "nframe": 16,
            "use_subtitle": true
        }
    }
}
```

Explanation of the config json:

1. Now we support two fields: `model` and `data`, each of which is a dictionary. The key of the `model` dictionary is the model name. The key of the `data` dictionary is an output alias. Output aliases must be safe filename components: use only letters, digits, `.`, `_`, and `-`, and start with a letter or digit.
2. For items in `model`, the value is a dictionary containing the following keys:
    - `class`: The class name of the model, which should be a class name defined in `vlmeval/vlm/__init__.py` (open-source models) or `vlmeval/api/__init__.py` (API models).
    - Other kwargs: Other kwargs are model-specific parameters, please refer to the definition of the model class for detailed usage. For example, `model`, `temperature`, `img_detail` are arguments of the `GPT4V` class. It's noteworthy that the `model` argument is required by most model classes.
    - Tip: The defined model in the `supported_VLM` of `vlmeval/config.py` can be used as a shortcut, for example, `GPT4o_20241120: {}` is equivalent to `GPT4o_20241120: {'class': 'GPT4V', 'model': 'gpt-4o-2024-11-20', 'temperature': 0, 'img_size': -1, 'img_detail': 'high', 'retry': 10, 'verbose': False}`
3. For the dictionary `data`, the key is an output alias. It only affects output-side names such as prediction files, status entries, reused artifacts, and symlinks. Dataset construction, post-processing, judge settings, and evaluation logic use the `dataset` field in the value. For items in `data`, the value is a dictionary containing the following keys:
    - `class`: The class name of the dataset, which should be a class name defined in `vlmeval/dataset/__init__.py`.
    - `dataset`: The logical dataset name passed to the dataset class constructor. It is required when `class` is set.
    - Other kwargs: Other kwargs are dataset-specific parameters, please refer to the definition of the dataset class for detailed usage. Typically, the `dataset` argument is required by most dataset classes. It's noteworthy that the `nframe` argument or `fps` argument is required by most video dataset classes.
    - Tip: Predefined video shortcuts can be launched directly with `--data MMBench_Video_8frame_nopack`. To reuse a shortcut under an output alias in config, write `my_alias: {'preset': 'MMBench_Video_8frame_nopack'}`. Empty dict shortcut configs such as `MMBench_Video_8frame_nopack: {}` are not supported.
Saving the example config json to `config.json`, you can launch the evaluation by:

```bash
python run.py --config config.json
```

That will generate the following output files under the working directory `$WORK_DIR` (Following the format `{$WORK_DIR}/{$MODEL_NAME}/{$MODEL_NAME}_{$DATA_ALIAS}_*`, where `$DATA_ALIAS` is the `data` key):

- `$WORK_DIR/GPT4o_20240806_T00_HIGH/GPT4o_20240806_T00_HIGH_MME-RealWorld-Lite*`
- `$WORK_DIR/GPT4o_20240806_T10_Low/GPT4o_20240806_T10_Low_MME-RealWorld-Lite*`
- `$WORK_DIR/GPT4o_20240806_T00_HIGH/GPT4o_20240806_T00_HIGH_MMBench_DEV_EN_V11*`
- `$WORK_DIR/GPT4o_20240806_T10_Low/GPT4o_20240806_T10_Low_MMBench_DEV_EN_V11*`
...
