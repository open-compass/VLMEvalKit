import json
import os
import shutil
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw

from .modules.latex2bbox_color import latex2bbox_color
from .modules.ransac import ransac
from .modules.visual_matcher import HungarianMatcher, SimpleAffineTransform


def gen_color_list(num=10, gap=15):
    num += 1
    single_num = 255 // gap + 1
    max_num = single_num**3
    num = min(num, max_num)
    color_list = []
    for idx in range(num):
        R = idx // single_num**2
        GB = idx % single_num**2
        G = GB // single_num
        B = GB % single_num

        color_list.append((R * gap, G * gap, B * gap))
    return color_list[1:]


_TOTAL_COLOR_LIST = None


def _display_path(path: str) -> str:
    abs_path = os.path.abspath(path)
    cwd = os.path.abspath(os.getcwd())
    try:
        rel_path = os.path.relpath(abs_path, cwd)
    except ValueError:
        return abs_path
    if rel_path == ".." or rel_path.startswith(f"..{os.sep}"):
        return abs_path
    return rel_path


def _copy_if_exists(src: str, dst: str) -> bool:
    if not os.path.exists(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _persist_cdm_visualization(
    *,
    tmp_root: str,
    basename: str,
    metrics: dict[str, float | int],
    persist_vis_dir: str | os.PathLike[str] | None,
    vis_name: str | None,
    vis_meta: dict[str, object] | None,
) -> dict[str, str]:
    persist_root = os.fspath(persist_vis_dir or os.path.join("result", "cdm_vis"))
    case_name = str(vis_name or f"{basename}_{os.getpid()}_{time.time_ns()}").strip()
    if not case_name:
        case_name = f"{basename}_{os.getpid()}_{time.time_ns()}"

    case_dir = os.path.join(persist_root, case_name)
    os.makedirs(case_dir, exist_ok=True)

    assets: dict[str, str] = {}
    file_map = {
        "match_png": (os.path.join(tmp_root, "vis_match", basename + ".png"), "match.png"),
        "match_base_png": (os.path.join(tmp_root, "vis_match", basename + "_base.png"), "match_base.png"),
        "gt_base_png": (os.path.join(tmp_root, "gt", "vis", basename + "_base.png"), "gt_base.png"),
        "gt_vis_png": (os.path.join(tmp_root, "gt", "vis", basename + ".png"), "gt_vis.png"),
        "pred_base_png": (os.path.join(tmp_root, "pred", "vis", basename + "_base.png"), "pred_base.png"),
        "pred_vis_png": (os.path.join(tmp_root, "pred", "vis", basename + ".png"), "pred_vis.png"),
        "gt_bbox_jsonl": (os.path.join(tmp_root, "gt", "bbox", basename + ".jsonl"), "gt_bbox.jsonl"),
        "pred_bbox_jsonl": (os.path.join(tmp_root, "pred", "bbox", basename + ".jsonl"), "pred_bbox.jsonl"),
    }
    for key, (src, file_name) in file_map.items():
        dst = os.path.join(case_dir, file_name)
        if _copy_if_exists(src, dst):
            assets[key] = _display_path(dst)

    meta_payload: dict[str, object] = {
        "case_name": case_name,
        "basename": basename,
        "case_dir": _display_path(case_dir),
        "metrics": metrics,
        "assets": assets,
    }
    if isinstance(vis_meta, dict):
        meta_payload.update(vis_meta)

    meta_path = os.path.join(case_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, ensure_ascii=False, indent=2)

    result = {
        "case_dir": _display_path(case_dir),
        "meta_json": _display_path(meta_path),
    }
    result.update(assets)
    return result


def _get_total_color_list():
    global _TOTAL_COLOR_LIST
    if _TOTAL_COLOR_LIST is None:
        _TOTAL_COLOR_LIST = gen_color_list(num=5800)
    return _TOTAL_COLOR_LIST


def update_inliers(ori_inliers, sub_inliers):
    inliers = np.copy(ori_inliers)
    sub_idx = -1
    for idx in range(len(ori_inliers)):
        if not ori_inliers[idx]:
            sub_idx += 1
            if sub_inliers[sub_idx]:
                inliers[idx] = True
    return inliers


def process_single_image(args, save_vis: bool = True):
    basename, gt_box_dir, pred_box_dir, match_vis_dir, max_iter, min_samples, residual_threshold, max_trials = args
    gt_valid, pred_valid = True, True
    if not os.path.exists(os.path.join(gt_box_dir, "bbox", basename + ".jsonl")):
        gt_valid = False
    else:
        with open(os.path.join(gt_box_dir, "bbox", basename + ".jsonl"), "r") as f:
            box_gt = []
            for line in f:
                info = json.loads(line)
                if info["bbox"]:
                    box_gt.append(info)
        if not box_gt:
            gt_valid = False
    if not gt_valid:
        return None

    if not os.path.exists(os.path.join(pred_box_dir, "bbox", basename + ".jsonl")):
        pred_valid = False
    else:
        with open(os.path.join(pred_box_dir, "bbox", basename + ".jsonl"), "r") as f:
            box_pred = []
            for line in f:
                info = json.loads(line)
                if info["bbox"]:
                    box_pred.append(info)
        if not box_pred:
            pred_valid = False
    if not pred_valid:
        return (
            basename,
            {
                "recall": 0.0,
                "precision": 0.0,
                "F1_score": 0.0,
                "tp": 0,
                "gt_tokens": int(len(box_gt)),
                "pred_tokens": 0,
            },
            None,
        )

    gt_img_path = os.path.join(gt_box_dir, "vis", basename + "_base.png")
    pred_img_path = os.path.join(pred_box_dir, "vis", basename + "_base.png")
    img_gt = Image.open(gt_img_path)
    img_pred = Image.open(pred_img_path)
    matcher = HungarianMatcher()
    matched_idxes = matcher(box_gt, box_pred, img_gt.size, img_pred.size)
    src = []
    dst = []
    for idx1, idx2 in matched_idxes:
        x1min, y1min, x1max, y1max = box_gt[idx1]["bbox"]
        x2min, y2min, x2max, y2max = box_pred[idx2]["bbox"]
        x1_c, y1_c = float((x1min + x1max) / 2), float((y1min + y1max) / 2)
        x2_c, y2_c = float((x2min + x2max) / 2), float((y2min + y2max) / 2)
        src.append([y1_c, x1_c])
        dst.append([y2_c, x2_c])
    src = np.array(src)
    dst = np.array(dst)
    if src.shape[0] <= min_samples:
        inliers = np.array([True for _ in matched_idxes])
    else:
        inliers = np.array([False for _ in matched_idxes])
        for _ in range(max_iter):
            remaining_mask = ~inliers
            if src[remaining_mask].shape[0] <= min_samples:
                break
            _, inliers_1 = ransac(
                (src[remaining_mask], dst[remaining_mask]),
                SimpleAffineTransform,
                min_samples=min_samples,
                residual_threshold=residual_threshold,
                max_trials=max_trials,
            )
            if inliers_1 is not None and inliers_1.any():
                inliers = update_inliers(inliers, inliers_1)
            else:
                break
            if np.count_nonzero(inliers) >= len(matched_idxes):
                break
    for idx, (a, b) in enumerate(matched_idxes):
        if inliers[idx] and matcher.cost["token"][a, b] == 1:
            inliers[idx] = False
    final_match_num = int(np.count_nonzero(inliers))
    recall = round(final_match_num / (len(box_gt)), 3)
    precision = round(final_match_num / (len(box_pred)), 3)
    F1_score = round(2 * final_match_num / (len(box_gt) + len(box_pred)), 3)
    metrics_per_img = {
        "recall": recall,
        "precision": precision,
        "F1_score": F1_score,
        "tp": int(final_match_num),
        "gt_tokens": int(len(box_gt)),
        "pred_tokens": int(len(box_pred)),
    }

    if save_vis:
        gap = 5
        W1, H1 = img_gt.size
        W2, H2 = img_pred.size
        H = H1 + H2 + gap
        W = max(W1, W2)
        vis_img = Image.new("RGB", (W, H), (255, 255, 255))
        vis_img.paste(img_gt, (0, 0))
        vis_img.paste(Image.new("RGB", (W, gap), (120, 120, 120)), (0, H1))
        vis_img.paste(img_pred, (0, H1 + gap))
        match_img = vis_img.copy()
        match_draw = ImageDraw.Draw(match_img)
        gt_matched_idx = {a: flag for (a, b), flag in zip(matched_idxes, inliers)}
        pred_matched_idx = {b: flag for (a, b), flag in zip(matched_idxes, inliers)}
        for idx, box in enumerate(box_gt):
            if idx in gt_matched_idx and gt_matched_idx[idx]:
                color = "green"
            else:
                color = "red"
            x_min, y_min, x_max, y_max = box["bbox"]
            match_draw.rectangle([x_min - 1, y_min - 1, x_max + 1, y_max + 1], fill=None, outline=color, width=2)
        for idx, box in enumerate(box_pred):
            if idx in pred_matched_idx and pred_matched_idx[idx]:
                color = "green"
            else:
                color = "red"
            x_min, y_min, x_max, y_max = box["bbox"]
            match_draw.rectangle(
                [x_min - 1, y_min - 1 + H1 + gap, x_max + 1, y_max + 1 + H1 + gap], fill=None, outline=color, width=2
            )
        os.makedirs(match_vis_dir, exist_ok=True)
        vis_img.save(os.path.join(match_vis_dir, basename + "_base.png"))
        match_img.save(os.path.join(match_vis_dir, basename + ".png"))

    return (basename, metrics_per_img, None)


def cdm_metrics(
    latex1: str,
    latex2: str,
    save_vis: bool = False,
    tmp_dir: str = "./tmp",
    persist_vis_dir: str | os.PathLike[str] | None = None,
    vis_name: str | None = None,
    vis_meta: dict[str, object] | None = None,
) -> dict[str, object]:
    """
    计算单条样本的 CDM 指标，返回 F1 与 token 统计。

    - latex1: GT LaTeX
    - latex2: Pred LaTeX
    - tmp_dir: 临时目录根路径（默认 ./tmp）
    - persist_vis_dir: save_vis=True 时的持久化目录，默认 result/cdm_vis
    """
    basename = "sample_0"

    # 确保 tmp_dir 存在（相对路径默认在当前工作目录下）
    os.makedirs(tmp_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cdm_calc_", dir=tmp_dir) as tmp_root:
        gt_box_dir = os.path.join(tmp_root, "gt")
        pred_box_dir = os.path.join(tmp_root, "pred")
        match_vis_dir = os.path.join(tmp_root, "vis_match")
        gt_temp_dir = os.path.join(tmp_root, "temp_gt")
        pred_temp_dir = os.path.join(tmp_root, "temp_pred")

        os.makedirs(os.path.join(gt_box_dir, "bbox"), exist_ok=True)
        os.makedirs(os.path.join(gt_box_dir, "vis"), exist_ok=True)
        os.makedirs(os.path.join(pred_box_dir, "bbox"), exist_ok=True)
        os.makedirs(os.path.join(pred_box_dir, "vis"), exist_ok=True)
        os.makedirs(gt_temp_dir, exist_ok=True)
        os.makedirs(pred_temp_dir, exist_ok=True)

        total_color_list = _get_total_color_list()
        latex2bbox_color((latex1, basename, gt_box_dir, gt_temp_dir, total_color_list))
        latex2bbox_color((latex2, basename, pred_box_dir, pred_temp_dir, total_color_list))

        max_iter = 5
        min_samples = 2
        residual_threshold = 30
        max_trials = 500

        result = process_single_image(
            (basename, gt_box_dir, pred_box_dir, match_vis_dir, max_iter, min_samples, residual_threshold, max_trials),
            save_vis=save_vis,
        )
        if result is None:
            return {
                "recall": 0.0,
                "precision": 0.0,
                "F1_score": 0.0,
                "tp": 0,
                "gt_tokens": 0,
                "pred_tokens": 0,
            }
        _basename, metrics, _ = result
        output: dict[str, object] = {
            "recall": float(metrics.get("recall", 0.0)),
            "precision": float(metrics.get("precision", 0.0)),
            "F1_score": float(metrics.get("F1_score", 0.0)),
            "tp": int(metrics.get("tp", 0)),
            "gt_tokens": int(metrics.get("gt_tokens", 0)),
            "pred_tokens": int(metrics.get("pred_tokens", 0)),
        }
        if save_vis:
            try:
                output["cdm_vis"] = _persist_cdm_visualization(
                    tmp_root=tmp_root,
                    basename=basename,
                    metrics={
                        "recall": float(output.get("recall", 0.0) or 0.0),
                        "precision": float(output.get("precision", 0.0) or 0.0),
                        "F1_score": float(output.get("F1_score", 0.0) or 0.0),
                        "tp": int(output.get("tp", 0) or 0),
                        "gt_tokens": int(output.get("gt_tokens", 0) or 0),
                        "pred_tokens": int(output.get("pred_tokens", 0) or 0),
                    },
                    persist_vis_dir=persist_vis_dir,
                    vis_name=vis_name,
                    vis_meta=vis_meta,
                )
            except Exception as exc:
                output["cdm_vis_error"] = f"{type(exc).__name__}: {exc}"
        return output


def cdm(latex1: str, latex2: str, save_vis: bool = False, tmp_dir: str = "./tmp") -> float:
    """
    计算单条样本的 CDM 分数（等价批量评测的 F1_score）。

    - latex1: GT LaTeX
    - latex2: Pred LaTeX
    - tmp_dir: 临时目录根路径（默认 ./tmp）
    """
    metrics = cdm_metrics(latex1, latex2, save_vis=save_vis, tmp_dir=tmp_dir)
    return float(metrics.get("F1_score", 0.0))


# Backward-compat alias
calc = cdm
