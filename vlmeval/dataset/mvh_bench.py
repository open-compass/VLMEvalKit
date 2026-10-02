import hashlib
import os
import os.path as osp
import random
import shutil

import pandas as pd

from .image_base import ImageBaseDataset


SHUFFLE_SEED = 2025


class MVHBench(ImageBaseDataset):

    # MVH-Bench mixes MCQ and binary QA.
    # Use a custom type so model-specific MCQ/VQA prompts do not override
    # the benchmark prompt.
    TYPE = "MVH"

    DATASET_URL = {
        "MVHBench": "None",
        "MVHBench_CrossInstance": "None",
        "MVHBench_CrossView": "None",
        "MVHBench_Val": "None",
    }

    @staticmethod
    def _get_extension(data):
        if data.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if data.startswith(b"\x89PNG"):
            return ".png"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return ".webp"
        return ".img"

    def _save_image(self, image, stem):
        raw = image.get("bytes")
        src_path = image.get("path")

        if raw is not None:
            ext = self._get_extension(raw)
            filename = f"{stem}{ext}"
            path = osp.join(self.img_root, filename)

            if not osp.exists(path):
                with open(path, "wb") as f:
                    f.write(raw)

            return filename

        if src_path is not None:
            ext = osp.splitext(src_path)[1] or ".jpg"
            filename = f"{stem}{ext}"
            path = osp.join(self.img_root, filename)

            if not osp.exists(path):
                shutil.copy2(src_path, path)

            return filename

        raise ValueError(f"Could not load image for {stem}")

    @staticmethod
    def _mc_choices(example, question_index):
        options = list(example["mc_options"])

        key = (
            f"{SHUFFLE_SEED}:"
            f"{example['group_id']}:"
            f"{question_index}"
        )
        seed = int.from_bytes(
            hashlib.sha256(key.encode()).digest(),
            "big",
        )
        random.Random(seed).shuffle(options)

        letters = "ABC"

        gold = letters[
            options.index(
                example["mc_answers"][question_index]
            )
        ]

        adversarial = letters[
            options.index(
                example["mc_answers"][1 - question_index]
            )
        ]

        neither = next(
            x for x in letters
            if x not in (gold, adversarial)
        )

        other = [
            x for x in letters
            if x != neither
        ]

        options[letters.index(neither)] = (
            f"Neither {other[0]} nor {other[1]}"
        )

        return options, gold, adversarial

    def load_data(self, dataset):
        from datasets import Image, load_dataset

        assert dataset in self.DATASET_URL

        split = (
            "validation"
            if dataset == "MVHBench_Val"
            else "test"
        )

        ds = load_dataset(
            "SNU-ISLAB/MVH-Bench",
            split=split,
        )

        if dataset == "MVHBench_CrossInstance":
            ds = ds.filter(
                lambda x: x == "cross_instance",
                input_columns=["mvh_type"],
            )

        elif dataset == "MVHBench_CrossView":
            ds = ds.filter(
                lambda x: x == "cross_view",
                input_columns=["mvh_type"],
            )

        os.makedirs(self.img_root, exist_ok=True)

        ds = ds.cast_column(
            "view_1",
            Image(decode=False),
        )
        ds = ds.cast_column(
            "view_2",
            Image(decode=False),
        )

        rows = []
        index = 0

        for example in ds:
            group_id = example["group_id"]

            view_1_path = self._save_image(
                example["view_1"],
                f"{group_id}_view1",
            )
            view_2_path = self._save_image(
                example["view_2"],
                f"{group_id}_view2",
            )

            # 2 multiple-choice questions
            for question_index in range(2):
                options, gold, adversarial = self._mc_choices(
                    example,
                    question_index,
                )

                rows.append(
                    {
                        "index": index,
                        "group_id": group_id,
                        "mvh_type": example["mvh_type"],
                        "category": example["category"],
                        "question_index": question_index,
                        "question_type": "mc",
                        "image_path": [
                            view_1_path,
                            view_2_path,
                        ],
                        "question": example["mc_questions"][
                            question_index
                        ],
                        "A": options[0],
                        "B": options[1],
                        "C": options[2],
                        "answer": gold,
                        "adversarial": adversarial,
                    }
                )
                index += 1

            # 4 binary questions
            for binary_index in range(4):
                question_index = binary_index + 2

                rows.append(
                    {
                        "index": index,
                        "group_id": group_id,
                        "mvh_type": example["mvh_type"],
                        "category": example["category"],
                        "question_index": question_index,
                        "question_type": "binary",
                        "image_path": [
                            view_1_path,
                            view_2_path,
                        ],
                        "question": example["binary_questions"][
                            binary_index
                        ],
                        "answer": example["binary_answers"][
                            binary_index
                        ],
                        "adversarial": "",
                    }
                )
                index += 1

        return pd.DataFrame(rows)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line)

        if line["question_type"] == "mc":
            choices = "\n".join(
                f"{letter}) {line[letter]}"
                for letter in "ABC"
            )

            prompt = (
                f"Question:\n{line['question']}\n\n"
                f"Choices:\n{choices}\n\n"
                "Only one option is correct.\n"
                "Present the answer strictly in the form X)."
            )

        else:
            prompt = (
                f"{line['question']}\n"
                "Please answer this question with one word."
            )

        return [
            dict(type="image", value=tgt_path[0]),
            dict(type="image", value=tgt_path[1]),
            dict(type="text", value=prompt),
        ]

    def evaluate(self, eval_file, **judge_kwargs):
        # Implemented in the next step.
        raise NotImplementedError