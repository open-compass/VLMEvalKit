import os
import os.path as osp
import shutil

import pandas as pd


from .image_mcq import ImageMCQDataset


class E3VQA(ImageMCQDataset):

    DATASET_URL = {
        "E3VQA": "None",
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

    def load_data(self, dataset):
        from datasets import Image, load_dataset

        assert dataset == "E3VQA"

        os.makedirs(self.img_root, exist_ok=True)

        ds = load_dataset(
            "SNU-ISLAB/E3VQA",
            split="test",
        )

        # Access the original encoded image bytes without decoding/re-encoding.
        ds = ds.cast_column("ego", Image(decode=False))
        ds = ds.cast_column("exo", Image(decode=False))

        rows = []

        for idx, example in enumerate(ds):
            options = example["options"]

            assert len(options) == 4
            assert example["answer"] in options

            answer_idx = options.index(example["answer"])
            answer_letter = "ABCD"[answer_idx]

            ego_path = self._save_image(
                example["ego"],
                f"{example['id']}_ego",
            )

            exo_path = self._save_image(
                example["exo"],
                f"{example['id']}_exo",
            )

            rows.append(
                {
                    "index": idx,
                    "id": example["id"],
                    "image_path": [ego_path, exo_path],
                    "question": example["question"],
                    "A": options[0],
                    "B": options[1],
                    "C": options[2],
                    "D": options[3],
                    "answer": answer_letter,
                    "source": example["source"],
                    "e3_category": example["category"],
                    "perspective": example["perspective"],
                    "l2-category": (
                        "egoexo4d"
                        if example["source"] == "Ego-Exo4D"
                        else "lemma"
                    ),
                    "category": f"{example['category']}_{example['perspective']}",
                }
            )

        return pd.DataFrame(rows)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line)

        options = [
            line["A"],
            line["B"],
            line["C"],
            line["D"],
        ]

        formatted_options = "\n".join(
            f"{letter}) {option}"
            for letter, option in zip("ABCD", options)
        )

        prompt = (
            f"Question:\n{line['question']}\n\n"
            f"Choices:\n{formatted_options}\n\n"
            "Only one option is correct.\n"
            "Present the answer in the form X).\n\n"
        )

        msgs = [
            dict(type="image", value=tgt_path[0]),
            dict(type="image", value=tgt_path[1]),
            dict(type="text", value=prompt),
        ]

        return msgs