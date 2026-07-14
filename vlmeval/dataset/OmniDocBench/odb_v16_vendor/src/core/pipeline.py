"""Structured pipeline entrypoints for the public evaluation runtime."""

import argparse
import io
import os
import pathlib

import yaml

from src.core.registry import DATASET_REGISTRY, EVAL_TASK_REGISTRY, load_default_registrations


def process_args(args):
    parser = argparse.ArgumentParser(description="Run the OmniDocBench public evaluation pipeline.")
    parser.add_argument("--config", "-c", type=str, default="./configs/end2end.yaml")
    return parser.parse_args(args)


def load_config(config_path):
    if isinstance(config_path, (str, pathlib.Path)):
        with io.open(os.path.abspath(config_path), "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise TypeError("Unexpected file type")

    if cfg is not None and not isinstance(cfg, (list, dict, str)):
        raise IOError(f"Invalid loaded object type: {type(cfg).__name__}")
    return cfg


def build_save_name(task_cfg):
    prediction_path = task_cfg["dataset"]["prediction"].get("data_path")
    if prediction_path:
        return os.path.basename(prediction_path) + "_" + task_cfg["dataset"].get("match_method", "quick_match")
    return os.path.basename(task_cfg["dataset"]["ground_truth"]["data_path"]).split(".")[0]


def run_config(cfg):
    load_default_registrations()

    for task_name in cfg.keys():
        if not cfg.get(task_name):
            print(f"No config for task {task_name}")
            continue

        task_cfg = cfg[task_name]
        dataset_name = task_cfg["dataset"]["dataset_name"]
        metrics_list = task_cfg["metrics"]
        val_dataset = DATASET_REGISTRY.get(dataset_name)(task_cfg)
        val_task = EVAL_TASK_REGISTRY.get(task_name)
        save_name = build_save_name(task_cfg)
        print("###### Process: ", save_name)

        page_info_path = task_cfg["dataset"]["ground_truth"].get("page_info")
        if page_info_path:
            val_task(val_dataset, metrics_list, page_info_path, save_name)
        else:
            val_task(val_dataset, metrics_list, task_cfg["dataset"]["ground_truth"]["data_path"], save_name)


def run_config_file(config_path):
    return run_config(load_config(config_path))


def main(argv=None):
    parameters = process_args(argv)
    run_config_file(parameters.config)


__all__ = [
    "build_save_name",
    "load_config",
    "main",
    "process_args",
    "run_config",
    "run_config_file",
]
