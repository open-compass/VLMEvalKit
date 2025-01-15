import abc
import json
import pathlib
from PIL import Image
import cv2
import math
import os
import logging
import tempfile
from utils import is_video_file
import re
from typing import Union, List, Dict


class BaseModel(abc.ABC):
    def __init__(
        self,
        api_key=None,
        model=None,
        query_data=None,
        resize=True,
        max_side=1000,
        print_response=False,
        max_num_image=None,
        system_message: Union[str, None] = None,
        total_demo_video_frames=4,
        **kwargs,
    ):
        self.api_key = api_key
        self.model = model
        self.query_data = query_data
        self.resize = resize
        self.max_side = max_side
        self.print_response = print_response
        self.prompts = self.load_prompts(
            pathlib.Path(__file__).resolve().parent / "prompt.json"
        )
        # the maximum number of images in each model API query
        self.max_num_image = max_num_image
        self.system_message = system_message
        self.total_demo_video_frames = total_demo_video_frames  # the number of frames sampled for videos for the demonstration examples
        # set number of max demo image to be the same as the number of frames per demo video
        self.max_demo_num_image = total_demo_video_frames

        # To be determined per task
        self.query_video_frames = None
        self.demo_video_frames = None

    @staticmethod
    def load_prompts(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def prepare_system_message(self) -> List[Dict[str, str]]:
        if self.system_message:
            return [{"role": "system", "content": self.system_message}]
        else:
            return []

    @abc.abstractmethod
    def prepare_context(self):
        pass

    @abc.abstractmethod
    def prepare_example_content(self, example_info):
        pass

    @abc.abstractmethod
    def prepare_query_content(self, query_info):
        pass

    @abc.abstractmethod
    def create_image_content(self, image_path):
        pass

    @staticmethod
    def _is_video_file(file_path):
        return is_video_file(file_path)

    def _set_sampling_config(self, query_idx):
        query_data = self.query_data
        num_query_videos = 0
        num_query_images = 0
        num_demo_videos = 0
        num_demo_images = 0
        for global_img in query_data["global_images"]:
            if self._is_video_file(global_img):
                num_query_videos += 1
            else:
                num_query_images += 1

        demo_example = query_data["example_info"]
        for demo_img in demo_example["image_paths"]:
            if self._is_video_file(demo_img):
                num_demo_videos += 1
            else:
                num_demo_images += 1

        query_example = query_data["queries"][query_idx]
        for query_img in query_example["image_paths"]:
            if self._is_video_file(query_img):
                num_query_videos += 1
            else:
                num_query_images += 1

        # the actual number of demo images to be used
        num_demo_images = min(self.max_demo_num_image, num_demo_images)

        if self.max_num_image:
            if num_demo_videos > 0:
                self.demo_video_frames = math.ceil(
                    self.total_demo_video_frames / num_demo_videos
                )
            else:
                self.demo_video_frames = 0

            if num_query_videos > 0:
                total_query_video_frames = (
                    self.max_num_image
                    - num_demo_images
                    - num_query_images
                    - self.demo_video_frames * num_demo_videos
                )
                if total_query_video_frames <= 0:
                    raise ValueError(
                        f"Cannot query <= 0 frames: please raise the number of maximum images allowed. {self.demo_video_frames=} {num_demo_videos=} {self.max_num_image=}"
                    )
                self.query_video_frames = total_query_video_frames // num_query_videos
            else:
                self.query_video_frames = 0

            total_num_image = (
                self.query_video_frames * num_query_videos
                + self.demo_video_frames * num_demo_videos
                + num_query_images
                + num_demo_images
            )
            exceed_image_quota = total_num_image > self.max_num_image
        else:
            exceed_image_quota = False

        return exceed_image_quota

    def process_video(self, video_path, is_demo):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

        # assert (
        #     "video_sampling" in self.query_data
        # ), "Missing video sampling strategy setting..."
        num_frames = self.demo_video_frames if is_demo else self.query_video_frames

        # the sampling rate using max number of frames
        sampling_gap_maxframe = (
            1 if not num_frames else math.ceil(frame_count / num_frames)
        )

        # If not set up, determine the sampling based on the video fps
        video_sampling = self.query_data.get("video_sampling", "fps")

        if video_sampling == "max":
            if fps >= 10:
                sampling_gap = max(math.ceil(fps / 5), sampling_gap_maxframe)
            else:
                sampling_gap = sampling_gap_maxframe
        elif video_sampling == "fps":
            sampling_gap_fps = (
                math.ceil(frame_count / self.demo_video_frames)
                if is_demo
                else math.ceil(fps)
            )
            sampling_gap = max(sampling_gap_fps, sampling_gap_maxframe)

        frame_number = 0
        images = []
        while True:
            success, frame = cap.read()
            if not success:
                break
            # Sample frames based on the dynamic sampling rate
            if frame_number % sampling_gap == 0:
                # Create a temporary file for the frame
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False
                ) as temp_frame:
                    cv2.imwrite(temp_frame.name, frame)
                    images.append(self.create_image_content(temp_frame.name))
                    os.remove(temp_frame.name)
            frame_number += 1
        if frame_number == 0:
            raise ValueError(f"Failed to read video from {video_path}, check data...")
        logging.info(
            f"Sampled {len(images)}/{frame_number} frames from video {video_path}"
        )
        cap.release()
        return images

    def create_media_content(self, file_path, is_demo=False):
        pass

    @staticmethod
    def _rgba_to_rgb(image):
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, image).convert("RGB")

    def _resize_image(self, image):
        resize_scale = self.max_side / max(image.size)
        new_size = (
            int(image.size[0] * resize_scale),
            int(image.size[1] * resize_scale),
        )
        return image.resize(new_size)

    def prepare_example_answer(self, example):
        assert isinstance(
            example["answers"], dict
        ), "Unexpected answers format, check the annotation!"
        answers = dict(example["answers"])  # make a copy of answers
        for field, result in answers.items():
            field_metric = self.query_data["fields_metrics"][field]
            if "multi_ref" in field_metric:
                # For multi-ref answers, only pick the first possible choice for the example answer
                from metrics.scoring.common.conversions import str_to_iterable

                refs = str_to_iterable(list, result)
                answers[field] = refs[0]

        if not self.use_cot:
            return (
                f"Answer: {list(answers.values())[0]}" if len(answers) == 1 else answers
            )
        else:
            return list(answers.values())[0] if len(answers) == 1 else answers

    @abc.abstractmethod
    def query(self, task_name, query_data, position=0):
        pass

    def clear(self):
        self.api_key = None
        self.model = None
        self.query_data = None
        self.resize = None
        self.max_side = None
        self.prompts = None

    def _process_text_and_media(self, text, media_paths, is_example=False):
        content = []
        chunks = re.split(r'(<image>|<video>)', text)

        placeholder_count = sum(1 for chunk in chunks if chunk in ['<image>', '<video>'])

        if placeholder_count != len(media_paths):
            raise ValueError(f"Mismatch between number of placeholders ({placeholder_count}) and media paths ({len(media_paths)})")

        media_index = 0
        curr_demo_images = 0
        for chunk in chunks:
            if chunk in ['<image>', '<video>']:
                media_content = self.create_media_content(media_paths[media_index], is_demo=is_example)
                if len(media_content) == 1:  # image
                    if is_example and curr_demo_images >= self.max_demo_num_image:
                        logging.warning("Exceed the quota for demo image, skip the demo image")
                    else:
                        content.extend(media_content)
                        if is_example:
                            curr_demo_images += 1
                else:  # video
                    content.extend(media_content)
                media_index += 1
            elif chunk.strip():  # Only add non-empty text chunks
                content.append({"type": "text", "text": chunk.strip()})

        return content
