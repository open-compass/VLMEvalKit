import json
import os
import shutil
import time
from datetime import datetime

import gradio as gr
import numpy as np
from modules.latex2bbox_color import latex2bbox_color
from modules.visual_matcher import HungarianMatcher, SimpleAffineTransform
from PIL import Image, ImageDraw
from skimage.measure import ransac

DATA_ROOT = "output"


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


def process_latex(groundtruths, predictions, user_id="test"):
    data_root = DATA_ROOT
    temp_dir = os.path.join(data_root, "temp_dir")

    data_root = os.path.join(data_root, user_id)
    output_dir_info = {}
    for subset, latex_list in zip(['gt', 'pred'], [groundtruths, predictions]):
        sub_temp_dir = os.path.join(temp_dir, f"{user_id}_{subset}")
        os.makedirs(sub_temp_dir, exist_ok=True)

        output_path = os.path.join(data_root, subset)
        output_dir_info[output_path] = []

        os.makedirs(os.path.join(output_path, 'bbox'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'vis'), exist_ok=True)

        total_color_list = gen_color_list(num=5800)

        for idx, latex in enumerate(latex_list):
            basename = f"sample_{idx}"
            input_arg = latex, basename, output_path, sub_temp_dir, total_color_list
            time.time()
            latex2bbox_color(input_arg)
            time.time()

    for subset in ['gt', 'pred']:
        shutil.rmtree(os.path.join(temp_dir, f"{user_id}_{subset}"))


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


def evaluation(user_id="test"):
    data_root = DATA_ROOT
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
    for basename in gt_basename_list:
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
            vis_img.paste(Image.new('RGB', (W, gap), (0, 150, 200)), (0, H1))
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
            if W < 500:
                padding = (500 - W) // 2 + 1
                reshape_match_img = Image.new('RGB', (500, H), (255, 255, 255))
                reshape_match_img.paste(match_img, (padding, 0))
                reshape_match_img.paste(Image.new('RGB', (500, gap), (0, 150, 200)), (0, H1))
                reshape_match_img.save(os.path.join(match_vis_dir, basename + ".png"))
            else:
                match_img.save(os.path.join(match_vis_dir, basename + ".png"))

    acc_list = [val['F1_score'] for _, val in metrics_per_img.items()]
    metrics_res = {
        "mean_score": round(np.mean(acc_list), 3),
        "details": metrics_per_img
    }
    metric_res_path = os.path.join(data_root, "metrics_res.json")
    with open(metric_res_path, "w") as f:
        f.write(json.dumps(metrics_res, indent=2))
    return metrics_res, metric_res_path, match_vis_dir


def calculate_metric_single(groundtruth, prediction):
    user_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    process_latex([groundtruth], [prediction], user_id)
    metrics_res, metric_res_path, match_vis_dir = evaluation(user_id)
    basename = "sample_0"
    image_path = os.path.join(match_vis_dir, basename + ".png")
    sample = metrics_res["details"][basename]
    score = sample['F1_score']
    recall = sample['recall']
    precision = sample['precision']
    return score, recall, precision, gr.Image(image_path)


def calculate_metric_batch(json_input):
    user_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    with open(json_input.name, "r") as f:
        input_data = json.load(f)
    groundtruths = []
    predictions = []
    for item in input_data:
        groundtruths.append(item['gt'])
        predictions.append(item['pred'])
    process_latex(groundtruths, predictions, user_id)
    metrics_res, metric_res_path, match_vis_dir = evaluation(user_id)
    return metric_res_path


def gradio_reset_single():
    return gr.update(value=None, placeholder='type gt latex code here'), gr.update(value=None, placeholder='type pred latex code here'), \
        gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None)


def gradio_reset_batch():
    return gr.update(value=None), gr.update(value=None)


def select_example1():
    gt = "y = 2x + 3z"
    pred = "y = 2z + 3x"
    return gr.update(value=gt, placeholder='type gt latex code here'), gr.update(
        value=pred, placeholder='type pred latex code here')


def select_example2():
    gt = "r = \\frac { \\alpha } { \\beta } \\vert \\sin \\beta \\left( \\sigma _ { 1 } \\pm \\sigma _ { 2 } \\right) \\vert"
    pred = "r={\\frac{\\alpha}{\\beta}}|\\sin\\beta\\left(\\sigma_{2}+\\sigma_{1}\\right)|"
    return gr.update(value=gt, placeholder='type gt latex code here'), gr.update(
        value=pred, placeholder='type pred latex code here')


def select_example3():
    gt = "\\begin{array} { r l r } & { } & { \\mathbf { J } _ { L } = \\left( \\begin{array} { c c } { 0 } & { 0 } \\\\ { v _ { n } } & { 0 } \\end{array} \\right) , ~ \\mathbf { J } _ { R } = \\left( \\begin{array} { c c } { u _ { n - 1 } } & { 0 } \\\\ { 0 } & { 0 } \\end{array} \\right) , ~ } \\\\ & { } & {\\mathbf { K } = \\left( \\begin{array} { c c } { V _ { n - 1 } } & { u _ { n } } \\\\ { v _ { n - 1 } } & { V _ { n } } \\end{array} \\right) , } \\end{array}"
    pred = "\\mathbf{J}_{U}={\\left(\\begin{array}{l l}{0}&{0}\\\\ {v_{n}}&{0}\\end{array}\\right)}\\,,\\ \\mathbf{J}_{R}={\\left(\\begin{array}{l l}{u_{n-1}}&{0}\\\\ {0}&{0}\\end{array}\\right)}\\,,\\mathbf{K}={\\left(\\begin{array}{l l}{V_{n-1}}&{u_{n}}\\\\ {v_{n-1}}&{V_{n}}\\end{array}\\right)}\\,,"
    return gr.update(value=gt, placeholder='type gt latex code here'), gr.update(
        value=pred, placeholder='type pred latex code here')


if __name__ == "__main__":
    title = """<h1 align="center">Character Detection Matching (CDM)</h1>"""

    with gr.Blocks() as demo:
        gr.Markdown(title)

        gr.Button(
            value="Quick Try: type latex code of gt and pred, get metrics and visualization.",
            interactive=False,
            variant="primary")

        with gr.Row():
            with gr.Column():
                gt_input = gr.Textbox(
                    label='gt',
                    placeholder='type gt latex code here',
                    interactive=True)
                pred_input = gr.Textbox(
                    label='pred',
                    placeholder='type pred latex code here',
                    interactive=True)
                with gr.Row():
                    clear_single = gr.Button("Clear")
                    submit_single = gr.Button(value="Submit", interactive=True, variant="primary")
                with gr.Accordion("Examples:"):
                    with gr.Row():
                        example1 = gr.Button("Example A(short)")
                        example2 = gr.Button("Example B(middle)")
                        example3 = gr.Button("Example C(long)")
            with gr.Column():
                with gr.Row():
                    score_output = gr.Number(label="F1 Score", interactive=False)
                    recall_output = gr.Number(label="Recall", interactive=False)
                    recision_output = gr.Number(label="Precision", interactive=False)
                gr.Button(
                    value="Visualization (green bbox means correcttlly matched, red bbox means missed or wrong.)",
                    interactive=False)
                vis_output = gr.Image(label=" ", interactive=False)

        example1.click(select_example1, inputs=None, outputs=[gt_input, pred_input])
        example2.click(select_example2, inputs=None, outputs=[gt_input, pred_input])
        example3.click(select_example3, inputs=None, outputs=[gt_input, pred_input])

        clear_single.click(
            gradio_reset_single,
            inputs=None,
            outputs=[
                gt_input,
                pred_input,
                score_output,
                recall_output,
                recision_output,
                vis_output])
        submit_single.click(
            calculate_metric_single, inputs=[
                gt_input, pred_input], outputs=[
                score_output, recall_output, recision_output, vis_output])

        gr.Button(
            value="Batch Run: upload a json file and batch processing, this may take some times according to your latex amount and length.",
            interactive=False,
            variant="primary")

        with gr.Row():
            with gr.Column():
                json_input = gr.File(label="Input Json", file_types=[".json"])
                json_example = gr.File(
                    label="Input Example",
                    value="assets/example/input_example.json")
                with gr.Row():
                    clear_batch = gr.Button("Clear")
                    submit_batch = gr.Button(value="Submit", interactive=True, variant="primary")

            metric_output = gr.File(label="Output Metrics")

        clear_batch.click(gradio_reset_batch, inputs=None, outputs=[json_input, metric_output])
        submit_batch.click(calculate_metric_batch, inputs=[json_input], outputs=[metric_output])

    demo.launch(share=True, server_name="0.0.0.0", server_port=10005, debug=True)
