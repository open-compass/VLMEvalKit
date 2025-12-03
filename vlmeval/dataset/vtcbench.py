import base64
import json

from .image_base import ImageBaseDataset

__doc__ = """
"""


def encode_image_bytes_to_base64(image_bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode()


class VTCBenchDataset(ImageBaseDataset):
    """
    VTCBench (Vision-Text Compression) Benchmark Dataset

    Including 3 subsets:
    - Retrieval: Image-Text Retrieval
    - Reasoning: Visual Reasoning
    - Memory: Long conversation with images
    """

    TYPE = "VQA"  # Use VQA type uniformly

    _DATASET_PATH = "MLLM-CL/VTCBench"

    # Dataset URL mapping - points to different splits of HuggingFace dataset
    DATASET_URL = {
        "Retrieval": f"https://huggingface.co/datasets/{_DATASET_PATH}",
        "Reasoning": f"https://huggingface.co/datasets/{_DATASET_PATH}",
        "Memory": f"https://huggingface.co/datasets/{_DATASET_PATH}",
    }

    # MD5 values are empty as HuggingFace datasets are dynamically loaded
    DATASET_MD5 = {
        "Retrieval": "",
        "Reasoning": "",
        "Memory": "",
    }

    def __init__(
        self, dataset: str = "Retrieval", skip_noimg: bool = False, language: str = "en"
    ):
        """Initialize dataset"""
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)
        self.language = language

    def load_data(self, dataset: str):
        """Load dataset from HuggingFace"""

        def _gen_fields(example: dict, idx: int) -> dict:
            # example schema:
            # problem: str
            # answers: list[str]
            # images: list[dict[str, bytes]] # bytes obj <=> jpeg image
            b64_imgs: list[str] = [
                encode_image_bytes_to_base64(img["bytes"]) for img in example["images"]
            ]
            return {
                "index": f"{dataset}_{idx}",
                "question": example["problem"],
                "answer": json.dumps(example["answers"], ensure_ascii=False),
                "image": json.dumps(b64_imgs),
                "category": dataset,
            }

        from datasets import load_dataset

        COLUMNS_ORGINIAL = ["problem", "answers", "images"]
        hf_dataset = load_dataset(
            self._DATASET_PATH, split=dataset, columns=COLUMNS_ORGINIAL
        )
        # apply transformation to VLMEval format
        hf_dataset = hf_dataset.map(
            _gen_fields,
            remove_columns=COLUMNS_ORGINIAL,
            with_indices=True,
            num_proc=16,
        )
        data = hf_dataset.to_pandas()
        # now data has schema:
        # index <class 'str'> Retrieval_0
        # question <class 'str'> What are all the special magic numbers for 019cc30e-2da8-4162-b145-df514e17 and demonic-heaven mentioned in the provided text?
        # answer <class 'str'> ["9199619", "1202641"]
        # image <class 'list'> [/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAx...XMBiHGRwbbBzgefJ8knJJ9ydXdTQTU1NTQTU1NTQTU1NTQTU1NTQTU1NTQTU1NTQTU1NTQTU1NTQTU1NTQTU1NTQTU1NTQf/2Q==, ...]
        # category <class 'str'> Retrieval

        return data


if __name__ == "__main__":
    dataset = VTCBenchDataset(dataset="Retrieval")
    print(f"Loaded {len(dataset)} samples from VTCBench Retrieval dataset.")
    sample = dataset[0]
    for k, v in sample.items():
        print(
            k,
            type(v),
            f"[{v[0][:100]}...{v[0][-100:]}, ...]" if isinstance(v, list) else v,
        )
