import os
import re
import tempfile
from functools import partial
from typing import List
import pandas as pd
import ast

from ..image_base import ImageBaseDataset, img_root_map
from ..utils import build_judge, DEBUG_MESSAGE
from ...smp import *
from ...utils import track_progress_rich
from ipdb import set_trace as st

logger = get_logger("RUN")

"""
{
    "img_filename": "web_3b0ad239-da6b-4f6f-8f12-f674dc90ff33.png",
    "bbox": [42, 1102, 197, 70],
    "question": "view the details of the item",
    "data_type": "text",
    "data_source": "shop"
},
{
    "img_filename": "web_3b0ad239-da6b-4f6f-8f12-f674dc90ff33.png",
    "bbox": [93, 74, 86, 132],
    "question": "view the previous photo",
    "data_type": "icon",
    "data_source": "shop"
}
"""

SYSTEM_PROMPT = """You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform pyautogui click/moveTo action to complete the task."""  # noqa: E501

USER_INSTRUCTION = """Please complete the following tasks by clicking using `pyautogui.click`:\n{instruction}"""

SYSTEM_PROMPT_V2 = """You are a GUI agent. You are given a screenshot of the screen and the description of a target element. You need to click the target element using `pyautogui.click`."""  # noqa: E501
USER_INSTRUCTION_V2 = """Please click the following target element using `pyautogui.click`:\n{description}"""


def parse_aguvis_coordinates(response_text):
    """Parse coordinates from model output string."""
    response_text = response_text.strip()
    response_text = (
        response_text.split("\n")[0]
        if len(response_text.split("\n")) > 1
        else response_text
    )

    if "pyautogui.click" in response_text or "pyautogui.moveTo" in response_text:
        coordinates = {}
        parts = response_text.split(",")
        for part in parts:
            if "x=" in part:
                coordinates["x"] = float(part.split("=")[1].strip())
            elif "y=" in part:
                coordinates["y"] = float(part.split("=")[1].strip().rstrip(")"))

        if "x" in coordinates and "y" in coordinates:
            return [
                coordinates["x"],
                coordinates["y"],
                coordinates["x"],
                coordinates["y"],
            ]
        else:
            print(f"Invalid coordinate format: {response_text}")
            return [0, 0, 0, 0]

    elif "wait" in response_text:
        return [-1, -1, -1, -1]
    else:
        print(f"Invalid format: {response_text}")
        return [0, 0, 0, 0]


class OSWorldG(ImageBaseDataset):
    MODALITY = "IMAGE"
    TYPE = "GUI"
    DATASET_URL = {
        "OSWorld-G": "OSWorld-G.tsv",
    }  # path
    DATASET_MD5 = {
        "OSWorld-G": "9a9be0b04a54fcf564b59b753622c42d",
    }
    EVAL_TYPE = "point"  # point or rectangle

    def __init__(
        self,
        dataset="OSWorld-G",
        skip_noimg=True,
        skeleton=False,
    ):
        # st()
        ROOT = LMUDataRoot()
        # You can override this variable to save image files to a different directory
        self.dataset_name = dataset
        self.img_root = osp.join(ROOT, "OSWorld-G", "images")
        if skeleton:
            return

        data = self.load_data(dataset)
        self.skip_noimg = skip_noimg
        if skip_noimg and "image" in data:
            data = data[~pd.isna(data["image"])]

        data["index"] = [str(idx + 1) for idx, x in enumerate(data["bbox"])]

        self.meta_only = True

        # The image field can store the base64 encoded image or another question index (for saving space)
        if "image" in data:
            data["image"] = [str(x) for x in data["image"]]
            image_map = {x: y for x, y in zip(data["index"], data["image"])}
            for k in image_map:
                if len(image_map[k]) <= 64:
                    idx = image_map[k]
                    assert idx in image_map and len(image_map[idx]) > 64
                    image_map[k] = image_map[idx]

            images = [toliststr(image_map[k]) for k in data["index"]]
            data["image"] = [x[0] if len(x) == 1 else x for x in images]
            self.meta_only = False

        if "img_filename" in data:
            paths = [toliststr(x) for x in data["img_filename"]]
            data["image_path"] = [x[0] if len(x) == 1 else x for x in paths]

        # if np.all([istype(x, int) for x in data["index"]]):
        #     data["index"] = [int(x) for x in data["index"]]

        self.data = data
        self.post_build(dataset)

    def prepare_tsv(self, url, file_md5=None):
        # breakpoint()
        data_root = LMUDataRoot()
        data_path = osp.join(data_root, "OSWorld-G", url)
        return pd.DataFrame(load(data_path))

    # actually retrieve the image path
    def dump_image(self, line):
        assert "image_path" in line
        tgt_path = toliststr(osp.join(self.img_root, line["image_path"]))
        return tgt_path

    @classmethod
    def get_action_space(self):
        return ""

    @classmethod
    def get_trajectory(self, line):
        traj_dict = {}
        traj_dict["task"] = line["question"]
        return traj_dict

    # get the actual classification list
    def get_classification_list(self, line):
        cls_list = line["classification_list"]
        cls_list = [x.strip() for x in cls_list[1:-1].split(",")]
        cls_list = [x[1:-1] for x in cls_list]
        return cls_list

    def get_float_list(self, line, col_name):
        bbox = line[col_name]
        bbox = [float(x) for x in bbox[1:-1].split(",")]
        return bbox

    def build_prompt(self, line):
        # st()
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)

        user_instruction = USER_INSTRUCTION.format(instruction=line["question"])

        msgs = []
        # add system prompt
        # if self.RE_TYPE == "functional":
        msgs.append(dict(role="system", type="text", value=SYSTEM_PROMPT))
        # else:
        #     msgs.append(dict(role="system", type="text", value=SYSTEM_PROMPT_V2))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="image", value=tgt_path)]
        msgs.append(dict(type="text", value=user_instruction))
        return msgs

    def _eval(
        self,
        coordinate: List[int],
        boxes_type: str,
        boxes_size: List[int],
        boxes_coordinate: List[int],
        image_size: List[int],
    ):

        def _is_point_in_rectangle(point, rect):
            return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]

        def _is_point_in_polygon(point, polygon):
            x, y = point
            n = len(polygon) // 2
            inside = False

            j = n - 1
            for i in range(n):
                xi, yi = polygon[i * 2], polygon[i * 2 + 1]
                xj, yj = polygon[j * 2], polygon[j * 2 + 1]

                if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                    inside = not inside
                j = i

            return inside

        # detect first if th coordiante are relative (between 0 and 1)
        if all(0 <= coord <= 1 for coord in coordinate):
            # expand the coordinate to the image width and height
            coordinate = [
                coord * image_size[i % 2] for i, coord in enumerate(coordinate)
            ]

        # get the center point of the predicted box
        center_x = (coordinate[0] + coordinate[2]) / 2
        center_y = (coordinate[1] + coordinate[3]) / 2
        center_point = [center_x, center_y]

        if boxes_type == "bbox":
            boxes_coordinate = [
                boxes_coordinate[0],
                boxes_coordinate[1],
                boxes_coordinate[0] + boxes_size[0],
                boxes_coordinate[1] + boxes_size[1],
            ]
            return _is_point_in_rectangle(center_point, boxes_coordinate)
        elif boxes_type == "polygon":
            return _is_point_in_polygon(center_point, boxes_coordinate)
        elif boxes_type == "refusal":
            # all the center point should be negative
            return all(center_point[i] < 0 for i in range(2))

    def evaluate(self, eval_file, **judge_kwargs):
        results_dict = {}

        result = []
        data = load(eval_file)

        assert "box_type" in data and "prediction" in data
        lt = len(data)
        correct = 0
        lines = [data.iloc[i] for i in range(lt)]
        for i in tqdm(range(len(lines))):
            line = lines[i]
            cls_list = self.get_classification_list(line)
            if len(cls_list) == 0:
                cls_list.append("unclassified")
                if "unclassified" not in results_dict:
                    results_dict["unclassified"] = {
                        "total": 0,
                        "correct": 0,
                        "accuracy": 0,
                    }
                results_dict["unclassified"]["total"] += 1
            else:
                for instance_group in cls_list:
                    if instance_group not in results_dict:
                        results_dict[instance_group] = {
                            "total": 0,
                            "correct": 0,
                            "accuracy": 0,
                        }
                    results_dict[instance_group]["total"] += 1

            prediction = str(line["prediction"])

            try:
                predicted_coords = parse_aguvis_coordinates(prediction)
            except Exception as e:
                logger.info(
                    f"Error parsing coordinates: {e}. The error response is: {prediction}"
                )
                predicted_coords = None
                continue

            bbox = self.get_float_list(line, "bbox")
            image_size = self.get_float_list(line, "image_size")

            if "bbox" == line["box_type"]:
                boxes_type = "bbox"
                boxes_coordinate = bbox[:2]
                boxes_size = bbox[2:]

            elif "polygon" == line["box_type"]:
                boxes_type = "polygon"
                boxes_coordinate = bbox
                boxes_size = image_size

            elif "refusal" == line["box_type"]:
                boxes_type = "refusal"
                boxes_coordinate = bbox
                boxes_size = image_size

            is_correct = self._eval(
                predicted_coords, boxes_type, boxes_size, boxes_coordinate, image_size
            )

            if is_correct:
                correct += 1
                for instance_group in cls_list:
                    results_dict[instance_group]["correct"] += 1

        accuracy = correct / lt
        for group in results_dict:
            results_dict[group][
                "accuracy"
            ] = f"{(results_dict[group]['correct'] / results_dict[group]['total']) * 100:.2f}%"

        results_dict = {
            "total": lt,
            "correct": correct,
            "accuracy": accuracy,
            "accuracy_dict_group": results_dict,
        }

        score_pth = eval_file.replace(".xlsx", "_score.json")
        dump(results_dict, score_pth)

        failure_cases_path = os.environ.get("FAILURE_CASES_PATH", None)
        if failure_cases_path is not None:
            failure_cases = [res for res in result if not res["match"] and res["is_wrong_format"]]
            failure_cases.sort(key=lambda r: r["num_matched"], reverse=True)

            with open(failure_cases_path, "w") as f:
                json.dump(failure_cases, f, indent=4, ensure_ascii=False)
        return results_dict
