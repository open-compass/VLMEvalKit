# flake8: noqa
import re
import os
import ast
import ipdb
import shutil
import zipfile
import subprocess
import vlmeval.dataset.utils.Ocrbench_v2.spotting_eval.rrc_evaluation_funcs_1_1 as rrc_evaluation_funcs


def extract_bounding_boxes_robust(predict_str):
    """
    Extract coordinates and text content from the given prediction string,
    handling potential format issues.

    Args:
    predict_str (str): Model prediction output as a string.

    Returns:
    list: Extracted data in the format [[x1, y1, x2, y2, text_content], ...].
          Returns None if no valid data is extracted.
    """
    results = []
    seen = set()

    # try parsing with ast.literal_eval
    try:
        data = ast.literal_eval(predict_str)
    except Exception:
        data = None

    if data is not None:
        if isinstance(data, (list, tuple)):
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) >= 5:
                    x1_str, y1_str, x2_str, y2_str = item[:4]
                    text_content = item[4]

                    x1_str = str(x1_str).strip()
                    y1_str = str(y1_str).strip()
                    x2_str = str(x2_str).strip()
                    y2_str = str(y2_str).strip()
                    text_content = str(text_content).replace("\n", "").strip().strip('"').strip("'")

                    try:
                        x1 = int(x1_str)
                        y1 = int(y1_str)
                        x2 = int(x2_str)
                        y2 = int(y2_str)

                        if not (0 <= x1 <= 1000 and 0 <= y1 <= 1000 and 0 <= x2 <= 1000 and 0 <= y2 <= 1000):
                            continue

                        key = (x1, y1, x2, y2, text_content)
                        if key in seen:
                            continue

                        seen.add(key)
                        results.append([x1, y1, x2, y2, text_content])
                    except ValueError:
                        continue
    else:
        # try parsing with regular expression

        list_content = predict_str
        items = re.findall(r'[\[\(]\s*([^\[\]\(\)]*?)\s*[\]\)]', list_content)

        if not items:
            return None

        for item in items:
            parts = item.split(',', 4)
            if len(parts) < 5:
                continue

            x1_str, y1_str, x2_str, y2_str, text_content = parts

            x1_str = x1_str.strip()
            y1_str = y1_str.strip()
            x2_str = x2_str.strip()
            y2_str = y2_str.strip()
            text_content = text_content.replace("\n", "").strip().strip('"').strip("'")

            try:
                x1 = int(x1_str)
                y1 = int(y1_str)
                x2 = int(x2_str)
                y2 = int(y2_str)

                if not (0 <= x1 <= 1000 and 0 <= y1 <= 1000 and 0 <= x2 <= 1000 and 0 <= y2 <= 1000):
                    continue

                key = (x1, y1, x2, y2, text_content)
                if key in seen:
                    continue

                seen.add(key)
                results.append([x1, y1, x2, y2, text_content])
            except ValueError:
                continue

    if not results:
        return None

    return results


def zip_folder(source_folder, destination_zip):
    abs_source = os.path.abspath(source_folder)
    abs_destination = os.path.abspath(destination_zip)

    with zipfile.ZipFile(abs_destination, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(abs_source):
            for file in files:
                abs_file_path = os.path.join(root, file)

                relative_path = os.path.relpath(abs_file_path, abs_source)
                zf.write(abs_file_path, relative_path)


def spotting_evaluation(prediction_list, img_metas):
    score = 0

    res_submit_list = []
    for item in prediction_list:
        if len(item) != 5:
            ipdb.set_trace()
        x1, y1, x2, y2, rec = item
        if x1 >= x2 or y1 >= y2:
            continue

        res_submit_list.append(",".join([str(x1),str(y1),str(x2),str(y1),str(x2),str(y2),str(x1),str(y2),rec]))

    res_gt_list = []
    for bbox, rec in zip(img_metas["bbox"], img_metas["content"]):
        x_coords = bbox[0::2]
        y_coords = bbox[1::2]

        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)

        res_gt_list.append(",".join([str(x1),str(y1),str(x2),str(y1),str(x2),str(y2),str(x1),str(y2),rec]))

    if len(res_submit_list) == 0 or len(res_gt_list) == 0:
        return 0

    command = {
        'g': res_gt_list,
        's': res_submit_list,
        'p': '{"IOU_CONSTRAINT":0.5}'
    }

    # run rrc_evaluation_funcs
    result = rrc_evaluation_funcs.main_evaluation(command)
    score = result["method"]["hmean"]
    return score
