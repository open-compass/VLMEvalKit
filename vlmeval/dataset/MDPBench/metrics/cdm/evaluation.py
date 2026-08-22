import argparse
import copy
import json
import os
import shutil
import time
from datetime import datetime
from multiprocessing import Pool

import numpy as np
from modules.latex2bbox_color import latex2bbox_color
from modules.visual_matcher import HungarianMatcher, SimpleAffineTransform
from PIL import Image, ImageDraw
from skimage.measure import ransac
from tqdm import tqdm


def gen_color_list(num=10, gap=15):
    num += 1
    single_num = 255 // gap + 1
    max_num = single_num ** 3
    num = min(num, max_num)
    color_list = []
    for idx in range(num):
        R = idx // single_num**2
        GB = idx % single_num**2
        G = GB // single_num
        B = GB % single_num

        color_list.append((R * gap, G * gap, B * gap))
    return color_list[1:]


def update_inliers(ori_inliers, sub_inliers):
    inliers = np.copy(ori_inliers)
    sub_idx = -1
    for idx in range(len(ori_inliers)):
        if ori_inliers[idx] == False:
            sub_idx += 1
            if sub_inliers[sub_idx] == True:
                inliers[idx] = True
    return inliers


def reshape_inliers(ori_inliers, sub_inliers):
    inliers = np.copy(ori_inliers)
    sub_idx = -1
    for idx in range(len(ori_inliers)):
        if ori_inliers[idx] == False:
            sub_idx += 1
            if sub_inliers[sub_idx] == True:
                inliers[idx] = True
        else:
            inliers[idx] = False
    return inliers


def gen_token_order(box_list):
    new_box_list = copy.deepcopy(box_list)
    for idx, box in enumerate(new_box_list):
        new_box_list[idx]['order'] = idx / len(new_box_list)
    return new_box_list


def evaluation(data_root, user_id="test"):
    data_root = os.path.join(data_root, user_id)
    gt_box_dir = os.path.join(data_root, "gt")
    pred_box_dir = os.path.join(data_root, "pred")
    match_vis_dir = os.path.join(data_root, "vis_match")
    os.makedirs(match_vis_dir, exist_ok=True)

    max_iter = 3
    min_samples = 3
    residual_threshold = 25
    max_trials = 50

    metrics_per_img = {}
    gt_basename_list = [item.split(".")[0]
                        for item in os.listdir(os.path.join(gt_box_dir, 'bbox'))]
    for basename in tqdm(gt_basename_list):
        gt_valid, pred_valid = True, True
        if not os.path.exists(os.path.join(gt_box_dir, 'bbox', basename + ".jsonl")):
            gt_valid = False
        else:
            with open(os.path.join(gt_box_dir, 'bbox', basename + ".jsonl"), 'r') as f:
                box_gt = []
                for line in f:
                    info = json.loads(line)
                    if info['bbox']:
                        box_gt.append(info)
            if not box_gt:
                gt_valid = False
        if not gt_valid:
            continue

        if not os.path.exists(os.path.join(pred_box_dir, 'bbox', basename + ".jsonl")):
            pred_valid = False
        else:
            with open(os.path.join(pred_box_dir, 'bbox', basename + ".jsonl"), 'r') as f:
                box_pred = []
                for line in f:
                    info = json.loads(line)
                    if info['bbox']:
                        box_pred.append(info)
            if not box_pred:
                pred_valid = False
        if not pred_valid:
            metrics_per_img[basename] = {
                "recall": 0,
                "precision": 0,
                "F1_score": 0,
            }
            continue
        gt_img_path = os.path.join(gt_box_dir, 'vis', basename + "_base.png")
        pred_img_path = os.path.join(pred_box_dir, 'vis', basename + "_base.png")

        img_gt = Image.open(gt_img_path)
        img_pred = Image.open(pred_img_path)

        matcher = HungarianMatcher()
        matched_idxes = matcher(box_gt, box_pred, img_gt.size, img_pred.size)
        src = []
        dst = []
        for (idx1, idx2) in matched_idxes:
            x1min, y1min, x1max, y1max = box_gt[idx1]['bbox']
            x2min, y2min, x2max, y2max = box_pred[idx2]['bbox']
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
            for i in range(max_iter):
                if src[inliers == False].shape[0] <= min_samples:
                    break
                model, inliers_1 = ransac((src[inliers == False], dst[inliers == False]), SimpleAffineTransform,
                                          min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials)
                if inliers_1 is not None and inliers_1.any():
                    inliers = update_inliers(inliers, inliers_1)
                else:
                    break
                if len(inliers[inliers == True]) >= len(matched_idxes):
                    break

        for idx, (a, b) in enumerate(matched_idxes):
            if inliers[idx] == True and matcher.cost['token'][a, b] == 1:
                inliers[idx] = False

        final_match_num = len(inliers[inliers == True])
        recall = round(final_match_num / (len(box_gt)), 3)
        precision = round(final_match_num / (len(box_pred)), 3)
        F1_score = round(2 * final_match_num / (len(box_gt) + len(box_pred)), 3)
        metrics_per_img[basename] = {
            "recall": recall,
            "precision": precision,
            "F1_score": F1_score,
        }

        if True:
            gap = 5
            W1, H1 = img_gt.size
            W2, H2 = img_pred.size
            H = H1 + H2 + gap
            W = max(W1, W2)

            vis_img = Image.new('RGB', (W, H), (255, 255, 255))
            vis_img.paste(img_gt, (0, 0))
            vis_img.paste(Image.new('RGB', (W, gap), (120, 120, 120)), (0, H1))
            vis_img.paste(img_pred, (0, H1 + gap))

            match_img = vis_img.copy()
            match_draw = ImageDraw.Draw(match_img)

            gt_matched_idx = {
                a: flag
                for (a, b), flag in
                zip(matched_idxes, inliers)
            }
            pred_matched_idx = {
                b: flag
                for (a, b), flag in
                zip(matched_idxes, inliers)
            }

            for idx, box in enumerate(box_gt):
                if idx in gt_matched_idx and gt_matched_idx[idx] == True:
                    color = "green"
                else:
                    color = "red"
                x_min, y_min, x_max, y_max = box['bbox']
                match_draw.rectangle([x_min - 1, y_min - 1, x_max + 1, y_max + 1],
                                     fill=None, outline=color, width=2)

            for idx, box in enumerate(box_pred):
                if idx in pred_matched_idx and pred_matched_idx[idx] == True:
                    color = "green"
                else:
                    color = "red"
                x_min, y_min, x_max, y_max = box['bbox']
                match_draw.rectangle([x_min - 1,
                                      y_min - 1 + H1 + gap,
                                      x_max + 1,
                                      y_max + 1 + H1 + gap],
                                     fill=None,
                                     outline=color,
                                     width=2)

            vis_img.save(os.path.join(match_vis_dir, basename + "_base.png"))
            match_img.save(os.path.join(match_vis_dir, basename + ".png"))

    score_list = [val['F1_score'] for _, val in metrics_per_img.items()]
    exp_list = [1 if score == 1 else 0 for score in score_list]
    metrics_res = {
        "mean_score": round(np.mean(score_list), 3),
        "exp_rate": round(np.mean(exp_list), 3),
        "details": metrics_per_img
    }
    metric_res_path = os.path.join(data_root, "metrics_res.json")
    with open(metric_res_path, "w") as f:
        f.write(json.dumps(metrics_res, indent=2))
    return metrics_res, metric_res_path, match_vis_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default="assets/example/input_example.json")
    parser.add_argument('--output', '-o', type=str, default="output")
    parser.add_argument('--pools', '-p', type=int, default=240)
    args = parser.parse_args()
    print(args)

    json_input, data_root, pool_num = args.input, args.output, args.pools
    temp_dir = os.path.join(data_root, "temp_dir")
    exp_name = os.path.basename(json_input).split('.')[0]
    with open(json_input, "r") as f:
        input_data = json.load(f)
    img_ids = []
    groundtruths = []
    predictions = []
    for idx, item in enumerate(input_data):
        if "img_id" in item:
            img_ids.append(item["img_id"])
        else:
            img_ids.append(f"sample_{idx}")
        groundtruths.append(item['gt'])
        predictions.append(item['pred'])

    a = time.time()
    user_id = exp_name

    total_color_list = gen_color_list(num=5800)

    data_root = os.path.join(data_root, user_id)
    output_dir_info = {}
    input_args = []
    for subset, latex_list in zip(['gt', 'pred'], [groundtruths, predictions]):
        sub_temp_dir = os.path.join(temp_dir, f"{exp_name}_{subset}")
        os.makedirs(sub_temp_dir, exist_ok=True)
        output_path = os.path.join(data_root, subset)
        output_dir_info[output_path] = []

        os.makedirs(os.path.join(output_path, 'bbox'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'vis'), exist_ok=True)

        for idx, latex in tqdm(enumerate(latex_list), desc=f"collect {subset} latex ..."):
            basename = img_ids[idx]
            input_arg = latex, basename, output_path, sub_temp_dir, total_color_list
            input_args.append(input_arg)

    if pool_num > 1:
        print(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "using processpool, pool num:",
            pool_num,
            ", job num:",
            len(input_args))
        myP = Pool(args.pools)
        for input_arg in input_args:
            myP.apply_async(latex2bbox_color, args=(input_arg,))
        myP.close()
        myP.join()
    else:
        for input_arg in input_args:
            latex2bbox_color(input_arg)
    b = time.time()
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
          "extract bbox done, time cost:", round(b - a, 3), "s")

    for subset in ['gt', 'pred']:
        shutil.rmtree(os.path.join(temp_dir, f"{exp_name}_{subset}"))

    c = time.time()
    metrics_res, metric_res_path, match_vis_dir = evaluation(args.output, exp_name)
    d = time.time()
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
          "calculate metrics done, time cost:", round(d - c, 3), "s")

    print(f"=> process done, mean f1 score: {metrics_res['mean_score']}.")
    print(f"=> more details of metrics are saved in `{metric_res_path}`")
    print(f"=> visulization images are saved under `{match_vis_dir}`")
