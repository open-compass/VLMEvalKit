from .image_base import ImageBaseDataset
from ..smp import load, dump, tqdm, d2df, np, osp, LMUDataRoot
from ast import literal_eval
import json, os, re, time

# We will use VLMEvalKit's OpenAI-compatible judge wrapper.
# It supports msgs = [{'type': 'text'|'image', 'value': ...}, ...]
from ..api.gpt import OpenAIWrapper

RUBRIC_SYSTEM = (
    "You are a rigorous evaluator of indoor scene captions. "
    "Score candidate captions for correctness, completeness, and faithfulness. "
    "Do NOT invent details not present in the image or references. "
    "Return STRICT JSON only."
)

# Default rubric focuses on indoor scenes
def make_user_prompt(pred_caption, ref_captions, wants_image=True, scale_max=10):
    return f"""
Judge the candidate caption for an indoor scene.

Candidate:
{pred_caption}

Reference captions (may contain multiple phrasings):
{json.dumps(ref_captions, ensure_ascii=False, indent=2)}

Scoring criteria (each 0–{scale_max}, integers preferred):
- room_type: Does it correctly identify the room type or setting (e.g., kitchen, office)?
- key_objects: Does it include salient indoor objects (e.g., sofa, table, bed) present in the scene?
- spatial_relations: Are important relations (e.g., "sofa against wall", "table under lamp") captured?
- cleanliness/illumination/style (optional bonus if depicted): Lighting conditions, tidiness, style cues.
- fluency: Is the caption grammatical and concise?
- hallucination_penalty: Deduct points for mentioning objects/attributes not present.

Compute:
- final_score: Weighted score on 0–{scale_max}. Recommended weights: room_type 0.25, key_objects 0.35, spatial_relations 0.25, fluency 0.15, minus hallucination_penalty.

Return JSON with keys exactly:
{{
  "room_type": <int 0-{scale_max}>,
  "key_objects": <int 0-{scale_max}>,
  "spatial_relations": <int 0-{scale_max}>,
  "fluency": <int 0-{scale_max}>,
  "hallucination_penalty": <int 0-{scale_max}>,
  "final_score": <int 0-{scale_max}>,
  "rationale": "<one short sentence>"
}}

Important:
- Output ONLY JSON (no markdown, no commentary).
- If uncertain, be conservative and penalize hallucinations.
""".strip()

class MMIS(ImageBaseDataset):
    TYPE = 'VQA'
    IMAGE_TYPE = "original" # or "distorted"

    @classmethod
    def supported_datasets(cls):
        return ['MMIS']

    def load_data(self, dataset):
        # Load from ~/LMUData/MMIS_original.tsv or MMIS_distorted.tsv
        data_path = osp.join(LMUDataRoot(), f'{dataset}_{MMIS.IMAGE_TYPE}.tsv')
        data = load(data_path)
        data['question'] = [(
            'Describe the image.'
        )] * len(data)
        return data

    @classmethod
    def evaluate(self, eval_file, **kwargs):
        """
        Args (through **kwargs via run.py --judge/--judge-args):
          model: judge model name, e.g., "gpt-4o" or "gpt-4o-mini"
          temperature: default 0
          max_tokens: default 256
          judge_args: dict-like, supports:
             - vision: bool (True = include the image for a vision judge; False = ref-only)
             - scale_max: int (default 10)
        """

        judge_model = kwargs.get('model', 'gpt-4o')
        temperature = kwargs.get('temperature', 0)
        max_tokens = kwargs.get('max_tokens', 256)
        vision = bool(kwargs.get('vision', True))
        scale_max = int(kwargs.get('scale_max', 10))

        judge = OpenAIWrapper(
            model=judge_model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=RUBRIC_SYSTEM,
        )

        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data, 'Need answer & prediction columns'
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]

        results = []
        num_fail = 0

        pbar = tqdm(range(len(lines)))
        for i in pbar:
            line = lines[i]
            pred = str(line['prediction']).strip()

            # "answer" column is a Python list string in VLMEvalKit caption datasets
            try:
                refs = literal_eval(str(line['answer']))
            except Exception:
                refs = [str(line['answer']).strip()]

            # Build multi-modal messages
            msgs = []
            if vision and 'image' in line and isinstance(line['image'], str) and os.path.exists(line['image']):
                msgs.append(dict(type='image', value=line['image']))
            msgs.append(dict(type='text', value=make_user_prompt(pred, refs, vision, scale_max)))

            # Ask judge for STRICT JSON; use JSON mode if server supports it
            ret, answer, _ = judge.generate_inner(
                msgs,
                response_format={"type": "json_object"}  # passed through in OpenAIWrapper
            )

            parsed = None
            if ret == 0:
                # Try strict JSON first
                try:
                    parsed = json.loads(answer)
                except Exception:
                    # Fallback: extract JSON blob if the model added extra text
                    m = re.search(r'\{.*\}', answer, re.S)
                    if m:
                        try:
                            parsed = json.loads(m.group(0))
                        except Exception:
                            parsed = None

            if not isinstance(parsed, dict) or 'final_score' not in parsed:
                num_fail += 1
                parsed = {
                    "room_type": 0, "key_objects": 0, "spatial_relations": 0,
                    "fluency": 0, "hallucination_penalty": 0,
                    "final_score": 0,
                    "rationale": "judge_error_or_invalid_json"
                }

            results.append({
                "index": i,
                "image": line.get('image', ''),
                "prediction": pred,
                "references": refs,
                "judge_model": judge_model,
                "score": parsed
            })

            if (i + 1) % 50 == 0:
                pbar.set_description(f"MMIS avg_so_far: {sum(x['score']['final_score'] for x in results)/(i+1):.3f}")

        # Aggregate
        avg = sum(x['score']['final_score'] for x in results) / max(1, len(results))
        out = {
            "metric": "LLM_JUDGE_INDOOR",
            "judge_model": judge_model,
            "scale_max": scale_max,
            "vision": vision,
            "avg_final_score": avg,
            "num_samples": len(results),
            "num_parse_fail": num_fail,
        }

        # Save alongside predictions
        base = eval_file.rsplit('.', 1)[0]
        dump(out, base + "_llmjudge_score.json")
        dump(results, base + "_llmjudge_items.jsonl")

        print(f"MMIS avg_final_score: {avg:.3f} (scale 0-{scale_max}); total={len(results)}; parse fails={num_fail}")
        return out