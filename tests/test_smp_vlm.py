import base64
import importlib.util
import io
from pathlib import Path

from PIL import Image, ImageFile


def _load_vlm_module():
    module_path = Path(__file__).resolve().parents[1] / "vlmeval" / "smp" / "vlm.py"
    spec = importlib.util.spec_from_file_location("vlmeval_smp_vlm_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _truncated_png_base64():
    image = Image.new("RGB", (20, 20), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()[:-20]).decode("utf-8")


def test_decode_base64_to_image_file_allows_recoverable_truncated_png(tmp_path):
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    vlm = _load_vlm_module()
    output_path = tmp_path / "recovered.png"

    vlm.decode_base64_to_image_file(_truncated_png_base64(), output_path)

    with Image.open(output_path) as recovered:
        recovered.load()
        assert recovered.size == (20, 20)
