import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from .base_metric import BaseMetric
from .runtime import get_available_cpu_count, get_metric_batch_size


class DINOScoreCalculator(BaseMetric):
    def __init__(self, batch_size=None):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.processor = self.get_DINOv2_model("base")
        self.model = self.model.to(self.device).eval()
        self.metric = self.calculate_DINOv2_similarity_score
        default_cpu_batch = max(8, min(get_available_cpu_count(), 32))
        self.batch_size = batch_size or get_metric_batch_size(
            "VLMEVAL_SARENA_DINO_BATCH_SIZE",
            cpu_default=default_cpu_batch,
            cuda_default=64,
            cpu_cap=32,
        )

    def get_DINOv2_model(self, model_size):
        model_map = {
            "small": "facebook/dinov2-small",
            "base": "facebook/dinov2-base",
            "large": "facebook/dinov2-large",
        }
        name = model_map.get(model_size)
        if not name:
            raise ValueError(f"model_size should be either 'small', 'base' or 'large', got {model_size}")

        model = AutoModel.from_pretrained(name)
        processor = AutoImageProcessor.from_pretrained(name)

        return model, processor

    def _ensure_pil_image(self, image):
        if isinstance(image, str):
            with Image.open(image) as pil_image:
                return pil_image.convert("RGB").copy()
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise ValueError("Input must be a file path or PIL Image")

    def process_input(self, image, processor):
        if isinstance(image, torch.Tensor):
            return image.unsqueeze(0) if image.dim() == 1 else image
        pil_image = self._ensure_pil_image(image)
        return self._encode_batch([pil_image])

    @torch.inference_mode()
    def _encode_batch(self, images):
        pil_images = [self._ensure_pil_image(image) for image in images]
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, non_blocking=True)
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state.mean(dim=1)

    def calculate_DINOv2_similarity_score(self, **kwargs):
        image1 = kwargs.get('gt_im')
        image2 = kwargs.get('pred_im')
        features1 = self.process_input(image1, self.processor)
        features2 = self.process_input(image2, self.processor)

        sim = F.cosine_similarity(features1, features2, dim=1).item()
        sim = (sim + 1) / 2

        return sim

    @torch.inference_mode()
    def calculate_score(self, batch, batch_size=None, update=True):
        gt_images = batch["gt_im"]
        pred_images = batch["pred_im"]

        effective_batch_size = batch_size or self.batch_size
        values = []

        for start in tqdm(
            range(0, len(gt_images), effective_batch_size),
            desc="DINO batches",
            leave=False,
        ):
            end = start + effective_batch_size
            features1 = self._encode_batch(gt_images[start:end])
            features2 = self._encode_batch(pred_images[start:end])
            sims = F.cosine_similarity(features1, features2, dim=1)
            sims = ((sims + 1.0) / 2.0).cpu().tolist()
            values.extend(sims)

        if not values:
            print("No valid values found for metric calculation.")
            return float("nan"), []

        avg_score = sum(values) / len(values)
        if update:
            self.meter.update(avg_score, len(values))
        return avg_score, values
