# FERMAT-PINK

FERMAT-PINK evaluates handwritten math OCR with **PINK**, from
[When VLMs 'Fix' Students: Identifying and Penalizing Over-Correction in the Evaluation of Multi-line Handwritten Math OCR](https://aclanthology.org/2026.emnlp-main.476/) (EMNLP 2026).
VLMs often "fix" the errors in a student's handwritten solution instead of transcribing
them. This is called over-correction, and string metrics such as BLEU barely notice it.

An LLM judge grades the transcription and the oracle transcription (what the student
actually wrote) with a 5-component, 100-point rubric. Components where the transcription
scores above the oracle are penalized (reverted if the gain is ≤ 10, set to 0 above).
PINK is the penalized sum divided by the oracle sum. A fully faithful transcription
scores 1.0.

```bash
pip install pink-eval
export OPENAI_API_KEY=...          # default judge: GPT-5, as in the paper
python run.py --data FERMAT_PINK --model Qwen2.5-VL-7B-Instruct --mode all
```

- **Data:** [FERMAT](https://huggingface.co/datasets/ai4bharat/FERMAT) (Nath et al., ACL 2025):
  2,244 handwritten solutions with known student errors, CC-BY-4.0. The dataset is gated,
  so accept its terms on the Hub and log in. The original image bytes are kept.
- **Judges:** the default `gpt-5` judge is scored against the paper's GPT-5 oracle grades,
  which ship with [`pink-eval`](https://github.com/9115jin/PINK), so results are comparable
  to the paper. For the open-source judge, serve Qwen3-Next-80B-A3B-Thinking (see the PINK
  repository) and pass
  `--judge qwen3-next-80b --judge-base-url http://localhost:8000/v1`; its oracle grades
  ship with `pink-eval` too. For any other `--judge`, the oracle transcriptions are graded
  as well. Set `PINK_ORACLE_GRADES` to reuse a grade file.
- **Outputs:** the score table (`PINK`, `over_correction_rate`, `raw_score_ratio`) is
  written as `<prediction-file-stem>_score.csv`. Judge results are cached beside the
  predictions, so an interrupted evaluation resumes.
- **`FERMAT_PINK_MINI`:** a fixed random subset of 100 samples for quick checks. Its
  scores are not comparable to full-set scores.
