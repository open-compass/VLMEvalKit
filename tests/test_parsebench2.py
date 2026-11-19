"""Native dataset contracts with the real pinned official scorer, no provider calls."""

import json

import pandas as pd
import pytest
from PIL import Image

from vlmeval.dataset import SUPPORTED_DATASETS, ParseBench2, build_dataset
from vlmeval.dataset import parsebench2_utils as pb2


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    for subset in pb2.SUBSETS:
        root = tmp_path / subset
        root.mkdir()
        image = Image.new("RGB", (100, 80), "white")
        image.save(root / "example.pdf", save_all=True, append_images=[image])
        reference = {"expected_markdown": "Hello **world**"}
        if subset == "layout":
            reference["test_rules"] = [
                {
                    "type": "layout",
                    "bbox": [0.1, 0.1, 0.5, 0.2],
                    "canonical_class": "Text",
                    "content": {"type": "text", "text": "Hello"},
                    "page": 1,
                }
            ]
        (root / "example.test.json").write_text(json.dumps(reference))
        (tmp_path / f"{subset}.jsonl").write_text(
            json.dumps(
                {
                    "id": f"{subset}/example",
                    "document": f"{subset}/example.pdf",
                    "expected_markdown": "SECRET",
                    "test_rules": ["SECRET"],
                }
            )
        )
    monkeypatch.setattr(pb2, "fetch", lambda name: str(tmp_path / name))
    (tmp_path / "lmu").mkdir()
    monkeypatch.setenv("LMUData", str(tmp_path / "lmu"))
    pb2.load_case.cache_clear()
    yield tmp_path
    pb2.load_case.cache_clear()


def predictions(path, rows):
    pd.DataFrame(rows, columns=["index", "prediction"]).to_csv(path, sep="\t", index=False)
    return str(path)


def test_native_registry_and_annotation_free_inputs(inputs):
    assert "ParseBench2" in SUPPORTED_DATASETS
    assert all(f"ParseBench2_{subset}" in SUPPORTED_DATASETS for subset in pb2.SUBSETS)
    dataset = build_dataset("ParseBench2")
    assert isinstance(dataset, ParseBench2)
    assert len(dataset) == 13
    assert set(dataset.data["subset"]) == set(pb2.SUBSETS)
    assert set(dataset.data.columns) == {"index", "example_id", "subset", "document", "question"}
    assert "SECRET" not in dataset.data.to_json()
    assert len(build_dataset("ParseBench2_forms")) == 1


def test_markdown_reference_survives_content_addressed_source(inputs, monkeypatch):
    # HF snapshots symlink source files to extensionless blobs. The official
    # loader resolves that path, so it cannot discover the companion itself.
    source = inputs / "text/example.pdf"
    blob = inputs / "blob"
    source.rename(blob)
    source.symlink_to(blob)
    (inputs / "text/example.test.json").write_text(json.dumps({"test_rules": []}))
    (inputs / "text/example.md").write_text("Official corrected reference")
    case = pb2.load_case("text/example.pdf")
    assert case.expected_markdown == "Official corrected reference"
    assert case.test_id == "text/example"
    doc = {"example_id": "text/example", "document": "text/example.pdf", "subset": "text"}
    item = pb2.score_document(doc, "Official corrected reference")
    assert item["evaluation"]["metrics"]


def test_native_prompt_all_pages_fixed_and_rgb(inputs):
    dataset = build_dataset("ParseBench2_text")
    messages = dataset.build_prompt(0)
    assert [message["type"] for message in messages] == ["image", "image", "text"]
    assert messages[-1]["value"] == pb2.PROMPT
    assert "SECRET" not in str(messages)
    for message in messages[:-1]:
        with Image.open(message["value"]) as image:
            assert image.mode == "RGB" and image.size == (200, 160)


@pytest.mark.parametrize("api_mode", [False, True])
def test_native_inference_cannot_replace_dataset_prompt(inputs, api_mode):
    from vlmeval.inference import infer_data
    from vlmeval.smp import load

    class Provider:
        is_api = api_mode

        def __init__(self):
            self.messages = []

        def set_dump_image(self, callback):
            self.dump_image_callback = callback

        def use_custom_prompt(self, dataset):
            return True

        def build_prompt(self, *args, **kwargs):
            raise AssertionError("Provider overrode the fixed benchmark prompt")

        def generate(self, message, dataset):
            self.messages.append(message)
            return "Hello **world**"

    dataset = build_dataset("ParseBench2_text")
    provider = Provider()
    output = inputs / "native-inference.pkl"
    infer_data(provider, "test-provider", str(inputs), dataset, str(output), api_nproc=1)
    assert load(str(output)) == {"text/example": "Hello **world**"}
    assert len(provider.messages) == 1
    assert [message["type"] for message in provider.messages[0]] == ["image", "image", "text"]
    assert provider.messages[0][-1]["value"] == pb2.PROMPT


def test_prediction_order_uses_ids_and_absent_rows_score_zero(inputs):
    dataset = build_dataset("ParseBench2_text")
    for suffix in ["pdf", "test.json"]:
        (inputs / f"text/other.{suffix}").write_bytes(
            (inputs / f"text/example.{suffix}").read_bytes()
        )
    other = {
        **dataset.data.iloc[0].to_dict(),
        "index": "text/other",
        "example_id": "text/other",
        "document": "text/other.pdf",
    }
    dataset.data = pd.concat([dataset.data, pd.DataFrame([other])], ignore_index=True)
    path = inputs / "predictions.tsv"
    good = dataset.evaluate(
        predictions(
            path,
            [
                ("text/other", "Hello **world**"),
                ("text/example", "Hello **world**"),
            ],
        )
    )
    missing = dataset.evaluate(predictions(path, [("text/other", "Hello **world**")]))
    assert good["text"] == 1
    assert missing["text"] == pytest.approx(0.5)
    assert missing["response_failure_rate"] == 0.5
    assert missing["scored_documents"] == 2
    assert missing["overall"] is None and missing["layout"] is None
    details = [
        json.loads(line)
        for line in (inputs / "predictions_parsebench2_details.jsonl").read_text().splitlines()
    ]
    assert len(details) == 2


@pytest.mark.parametrize(
    "response", ["", "{broken", '{"markdown":null}', "Failed to obtain answer"]
)
def test_native_missing_invalid_failures_score_zero(inputs, response):
    dataset = build_dataset("ParseBench2_text")
    result = dataset.evaluate(predictions(inputs / "failure.tsv", [("text/example", response)]))
    assert result["text"] == 0
    assert result["response_failure_rate"] == 1


@pytest.mark.parametrize(
    "rows,match",
    [
        ([("text/example", "x"), ("text/example", "y")], "Duplicate"),
        ([("text/unknown", "x")], "Unknown"),
    ],
)
def test_native_rejects_ambiguous_ids(inputs, rows, match):
    dataset = build_dataset("ParseBench2_text")
    with pytest.raises(ValueError, match=match):
        dataset.evaluate(predictions(inputs / "bad.tsv", rows))
    assert not (inputs / "bad_score.json").exists()


def test_saved_provider_record_preserved_and_identity_checked(inputs):
    doc = pb2.load_inputs()[0]
    payload = pb2._native_result(doc, {"markdown": "Hello", "layout_pages": []}).model_dump(
        mode="json"
    )
    payload["raw_output"] = {"provider_private_shape": {"pages": [1, 2]}}
    restored, supported, status = pb2.decode_response(doc, json.dumps(payload))
    assert restored.model_dump(mode="json") == payload
    assert supported and status == "official_record"
    payload["request"]["example_id"] = "text/wrong"
    with pytest.raises(ValueError, match="IDs disagree"):
        pb2.decode_response(doc, json.dumps(payload))


def test_mixed_missing_layout_retains_official_zero(inputs):
    dataset = build_dataset("ParseBench2_layout")
    doc = dataset.data.iloc[0].to_dict()
    payload = {
        "markdown": "Hello",
        "layout_pages": [
            {
                "page_number": 1,
                "width": 1,
                "height": 1,
                "items": [
                    {
                        "type": "text",
                        "md": "Hello",
                        "bbox": {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.2},
                    }
                ],
            }
        ],
    }
    good = pb2.score_document(doc, json.dumps(payload))
    bad = pb2.score_document(doc, None)
    bad["evaluation"]["example_id"] = bad["evaluation"]["test_id"] = "layout/other"
    score = pb2.combined_report([good])["dimensions"]["layout"]["score"]
    assert score > 0
    assert pb2.combined_report([good, bad])["dimensions"]["layout"]["score"] == pytest.approx(
        score / 2
    )
    assert pb2.combined_report([bad])["dimensions"]["layout"]["score"] is None


@pytest.mark.parametrize(
    "rules", [[], [{"type": "is_code_block", "language": "fortran", "code": "PRINT *, 181"}]]
)
def test_ineligible_reference_stops_native_score_output(inputs, rules):
    (inputs / "text/example.test.json").write_text(
        json.dumps({"expected_markdown": None, "test_rules": rules})
    )
    dataset = build_dataset("ParseBench2_text")
    with pytest.raises(RuntimeError, match="reference/scorer eligibility"):
        dataset.evaluate(predictions(inputs / "ineligible.tsv", [("text/example", "Hello")]))
    assert not (inputs / "ineligible_score.json").exists()


def test_cached_empty_official_metrics_cannot_be_aggregated(inputs):
    doc = pb2.load_inputs()[0]
    item = pb2.score_document(doc, "Hello")
    item["evaluation"]["metrics"] = []
    with pytest.raises(RuntimeError, match="reference/scorer eligibility"):
        pb2.combined_report([item])


@pytest.mark.parametrize("failure", ["reference", "duplicate_ids"])
def test_failed_rerun_removes_stale_score(inputs, failure):
    dataset = build_dataset("ParseBench2_text")
    path = predictions(inputs / "rerun.tsv", [("text/example", "Hello")])
    dataset.evaluate(path)
    assert (inputs / "rerun_score.json").exists()
    if failure == "reference":
        (inputs / "text/example.test.json").write_text(json.dumps({"test_rules": []}))
        pb2.load_case.cache_clear()
        error, match = RuntimeError, "reference/scorer eligibility"
    else:
        predictions(path, [("text/example", "Hello"), ("text/example", "Hello")])
        error, match = ValueError, "Duplicate"
    with pytest.raises(error, match=match):
        dataset.evaluate(path)
    assert not (inputs / "rerun_score.json").exists()
    assert not (inputs / "rerun_parsebench2_details.jsonl").exists()
