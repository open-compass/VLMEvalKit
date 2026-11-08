# VisualOverload

[VisualOverload](https://github.com/paulgavrikov/visualoverload) (CVPR 2026)
contains 2,720 questions about dense, high-resolution paintings. It combines
multiple-choice, counting, and OCR questions; all ground-truth answers are private.

```bash
python run.py --data VisualOverload --model Qwen2.5-VL-3B-Instruct --mode all
```

The dataset adapter downloads the public Hugging Face `test` split, retains the
original image bytes, and uses each record's `default_prompt` unchanged. Model
processors may still resize or tile inputs; report those settings with results.

The runner writes `<prediction-file-stem>_submission.json` beside the prediction
table, in both `all` and `infer` modes. The JSON contains a list of
`{"question_id": "...", "response": "..."}` records. Raw predictions, including
free-form counts and OCR answers, are preserved for official answer extraction.

Upload the file to the [official evaluator](https://huggingface.co/spaces/paulgavrikov/visualoverload-submit)
(Hugging Face sign-in required). No local accuracy or LLM judge is used and no
automatic submission is made. The run status records `external_submission_required`.

To export an existing prediction table directly:

```python
from vlmeval.dataset import VisualOverload

VisualOverload.export_submission('model_VisualOverload.xlsx')
```
