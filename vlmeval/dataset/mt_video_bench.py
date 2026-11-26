"""MT-Video-Bench: multi-turn video dialogue benchmark.

Paper: https://arxiv.org/abs/2510.17722
Official code: https://github.com/NJU-LINK/MT-Video-Bench
    (pinned to commit 547fe24720bd66e028e49b2d2bd8f95bea9c59c0)
Official data: https://huggingface.co/datasets/NJU-LINK/MT-Video-Bench (gated)
    (pinned to revision a8ca8142279c29dd7de73ddd67781c2185f2bb0e)

Protocol (``eval/infer_qwen25vl.py`` + ``eval/eval.py`` of the official repo):

* Every dialogue (one ``event_index`` of one video in one ability file) is a
  sequence of rounds. Each round is answered by an independent single-turn
  call whose user message is a text prompt that embeds the earlier rounds as a
  ``["User: ...", "Assistant: ...", ...]`` list, followed by the video.
* ``golden`` (official default): the embedded ``Assistant`` turns are the
  annotated reference answers. ``self_predicted`` (paper Fig. 8; no official
  inference code): they are the evaluated model's own earlier answers.
  The two conditions are separate dataset names (one row = one case).
* The video is sampled once (uniform ``np.linspace`` over at most ``nframe``
  frames, at native resolution) and the frames are reused by every round and
  every condition. Resolution / pixel-budget handling is left to the model.
* A checklist judge (official default ``gemini-2.5-flash``, 3 runs per round)
  scores each checklist round; round -> event -> ability -> overall means.
"""
import ast
import json
import os
import os.path as osp
import threading

import numpy as np
import pandas as pd
import portalocker
from PIL import Image

from vlmeval.smp import dump, get_intermediate_file_path, load
from vlmeval.smp.file import LMUDataRoot
from vlmeval.utils import track_progress_rich
from ..interactive_mt import Step, TurnPolicy, has_failure
from .interactive_mt import InteractiveMTMixin
from .utils.judge_util import build_judge
from .video_base import VideoBaseDataset

HF_REPO = 'NJU-LINK/MT-Video-Bench'
HF_REVISION = 'a8ca8142279c29dd7de73ddd67781c2185f2bb0e'

ABILITY_FILES = [
    'answer_refusal', 'memory_recall', 'object_reference',
    'proactive_interaction', 'summary', 'topic_shifting',
]
# Official ``ABILITY_LIST`` / ``ABILITY_1`` of infer_qwen25vl.py.
INFER_ROUND_ABILITIES = ('Summary', 'Object Reference', 'Memory Recall', 'Answer Refusal')
ALWAYS_INFER_FILES = ('proactive_interaction', 'topic_shifting')
NO_INFER_RESPONSE = 'No need for this ability to answer.'

INFER_PROMPT = (
    "You're an AI assistant helping to answer questions about a video. "
    "Use the conversation history if helpful.\n"
    "Conversation history:\n"
    "{history}\n"
    "Current question:\n"
    "{question}"
)

# Verbatim copy of the f-string in eval.py (including its 4-space indentation).
# Split only to satisfy the line-length limit; the assembled string is byte-identical.
JUDGE_PROMPT = """
    Given a current model-generated answer and a checklist of five questions, \
output whether the model has correctly addressed the above questions.

    ### Model Answer:
    {model_answer}

    ### Checklist:
    {checklist_str}

    **Output Format Example:**
    ```json
    {{
    "A1": "Yes/No",
    "A2": "Yes/No",
    "A3": "Yes/No",
    "A4": "Yes/No",
    "A5": "Yes/No"
    }}
    ```
    """

CONDITIONS = ('golden', 'self_predicted')


def extract_json(text):
    """Same as the official eval.py: first fenced block, else the raw text."""
    import regex as re
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1)
    return text


def _round_index(round_key):
    return int(str(round_key).split('Round ')[-1])


class MTVideoBenchPolicy(TurnPolicy):
    """Replay the rounds of one dialogue; one policy per (dialogue, condition)."""

    def __init__(self, dataset, case, first_round, video_llm=False):
        super().__init__()
        self.dataset = dataset
        self.case = case
        self.video_llm = video_llm
        self.rounds = dataset.rounds_of(case)
        self.infer_rounds = [i for i in range(len(self.rounds))
                             if dataset.round_needs_inference(case, i)]
        self.max_turns = len(self.infer_rounds)
        self.cur = first_round
        # None = not produced (yet). Skipped rounds get the official placeholder.
        self.state['responses'] = [None] * len(self.rounds)
        for i in range(first_round):
            self.state['responses'][i] = NO_INFER_RESPONSE

    def _history_answers(self):
        """Answers shown in the next history: reference (golden) or the model's own."""
        answers = []
        for i, rnd in enumerate(self.rounds):
            resp = self.state['responses'][i]
            own_answer = (
                self.case['condition'] == 'self_predicted'
                and i in self.infer_rounds and resp is not None)
            if own_answer:
                answers.append(resp)
            else:
                answers.append(rnd['Assistant'])
        return answers

    def on_failure(self, message, history, turn, reason='model'):
        self.state['responses'][self.cur] = message
        self.state['failure'] = {'message': message, 'turn': turn, 'reason': reason,
                                 'round': self.cur + 1}

    def step(self, assistant_response, history):
        self.state['responses'][self.cur] = assistant_response
        later = [i for i in self.infer_rounds if i > self.cur]
        end = later[0] if later else len(self.rounds)
        for i in range(self.cur + 1, end):
            self.state['responses'][i] = NO_INFER_RESPONSE
        if not later:
            return Step(done=True)
        self.cur = later[0]
        # The next call sees a freshly built context (video + embedded history of the
        # earlier rounds) instead of a growing chat, exactly like the official adapter.
        return Step(history=self.dataset.round_messages(
            self.case, self.cur, self._history_answers(), video_llm=self.video_llm))

    def finalize(self):
        return {
            'dialogue_id': self.case['dialogue_id'],
            'condition': self.case['condition'],
            'responses': list(self.state['responses']),
            'failure': self.state.get('failure'),
        }


class MTVideoBench(InteractiveMTMixin, VideoBaseDataset):

    TYPE = 'Video-VQA'
    MODALITY = 'VIDEO'
    DEFAULT_JUDGE_MODEL = 'gemini-2.5-flash'
    DEFAULT_POLICY_MODEL = None

    # dataset name -> history condition. Official reporting: golden is the main
    # setting; self-predicted is a separate ablation (paper Fig. 8), so each is
    # its own dataset name and result table (one row = one case).
    NAME_CONDITIONS = {
        'MT-Video-Bench': 'golden',
        'MT-Video-Bench-SelfPredicted': 'self_predicted',
    }

    def __init__(self, dataset='MT-Video-Bench', nframe=128, fps=-1,
                 judge_rounds=3, pack=False):
        if fps is not None and fps > 0:
            raise ValueError(
                'MT-Video-Bench samples at most `nframe` frames uniformly; fps is not supported.')
        if pack:
            raise ValueError('MT-Video-Bench does not support pack mode.')
        super().__init__(dataset=dataset, pack=False, nframe=nframe, fps=-1)
        # All variants share one frame cache.
        self.frame_root = osp.join(LMUDataRoot(), 'images', 'MT-Video-Bench')
        os.makedirs(self.frame_root, exist_ok=True)
        self.judge_rounds = judge_rounds
        self.condition = self.NAME_CONDITIONS[dataset]
        self._frames = {}
        self._frames_lock = threading.Lock()
        self._videos = {}
        self._video_lock = threading.Lock()
        self._json_cache = {}

    @classmethod
    def supported_datasets(cls):
        return list(cls.NAME_CONDITIONS)

    # ------------------------------------------------------------------ data

    def prepare_dataset(self, dataset):
        root = osp.join(LMUDataRoot(), 'MT-Video-Bench')
        data_file = osp.join(root, 'MT-Video-Bench.tsv')
        video_root = osp.join(root, 'videos')
        os.makedirs(video_root, exist_ok=True)

        def fetch(rel):
            local = osp.join(root, rel)
            if osp.exists(local) and os.path.getsize(local) > 0:
                return local
            try:
                from huggingface_hub import hf_hub_download
                hf_hub_download(HF_REPO, rel, repo_type='dataset', revision=HF_REVISION,
                                local_dir=root)
            except Exception as err:
                raise RuntimeError(
                    f'Cannot obtain {rel} of {HF_REPO}@{HF_REVISION[:8]}. The dataset is gated: '
                    f'request access at https://huggingface.co/datasets/{HF_REPO} (or contact the '
                    f'authors), authenticate with `huggingface-cli login`/HF_TOKEN, or place the '
                    f'files under {root} manually. '
                    f'Original error: {type(err).__name__}: {err}') from err
            return local

        if not osp.exists(data_file):
            rows = []
            for ability in ABILITY_FILES:
                mtqa = load(fetch(f'mtqa/{ability}.json'))
                checklist_data = load(fetch(f'checklist/{ability}_CheckList.json'))
                for video, events in mtqa.items():
                    for event in events:
                        event_index = str(event['event_index'])
                        rounds = event['mtqa']
                        first_key = next(iter(rounds))
                        checklist = checklist_data.get(video, {}).get(event_index)
                        rows.append(dict(
                            ability=ability, video=video, event_index=event_index,
                            question=rounds[first_key]['User'],
                            mtqa=json.dumps(rounds, ensure_ascii=False),
                            checklist=json.dumps(checklist, ensure_ascii=False),
                        ))
            frame = pd.DataFrame(rows)
            frame.insert(0, 'index', np.arange(len(frame)))
            dump(frame, data_file)

        # Videos are fetched lazily (one 15 GB collection; only the rows being run are needed).
        self._fetch = fetch
        return dict(root=video_root, data_file=data_file)

    # ------------------------------------------------------- row / case access

    def _row(self, sample):
        if isinstance(sample, (int, np.integer)):
            return dict(self.data.iloc[int(sample)])
        return dict(sample)

    def _json(self, row, key):
        raw = row[key]
        if isinstance(raw, str):
            ck = (row['index'], key)
            if ck not in self._json_cache:
                self._json_cache[ck] = json.loads(raw)
            return self._json_cache[ck]
        return raw

    def rounds_of(self, case):
        rounds = self._json(case['row'], 'mtqa')
        return [dict(rounds[key], key=key) for key in rounds]

    def round_needs_inference(self, case, i):
        row = case['row']
        if row['ability'] in ALWAYS_INFER_FILES:
            return True
        return self.rounds_of(case)[i].get('Ability') in INFER_ROUND_ABILITIES

    @staticmethod
    def dialogue_id(row):
        return f"{row['ability']}/{row['video']}/{row['event_index']}"

    def _case(self, sample):
        """Internal view of one dataset row (= one runner case) plus its condition.

        ``sample`` is a row position (what the runners pass to ``build_prompt``)
        or the row itself (what they pass to ``build_policy``).
        """
        row = self._row(sample)
        return dict(dialogue_id=self.dialogue_id(row), condition=self.condition, row=row)

    # ----------------------------------------------------------------- prompts

    def round_messages(self, case, i, answers, video_llm=False):
        rounds = self.rounds_of(case)
        items = []
        for j in range(i):
            items.append(f"User: {rounds[j]['User']}")
            items.append(f"Assistant: {answers[j]}")
        prompt = INFER_PROMPT.format(history=items, question=rounds[i]['User'])
        content = [dict(type='text', value=prompt)]
        if video_llm:
            content.append(dict(type='video', value=self.save_video_mp4(case['row']['video'])))
        else:
            frames = self.save_video_frames(case['row']['video'])
            content += [dict(type='image', value=path) for path in frames]
        return [dict(role='user', content=content)]

    def _first_inference_round(self, case):
        for i in range(len(self.rounds_of(case))):
            if self.round_needs_inference(case, i):
                return i
        raise ValueError(f"No inferable round in dialogue {case['dialogue_id']}.")

    def build_prompt(self, line, video_llm):
        # Image-only endpoints receive the sampled frames; native-video models get the
        # same frames re-encoded as a 15fps mp4, matching the official Qwen adapter.
        case = self._case(line)
        first = self._first_inference_round(case)
        rounds = self.rounds_of(case)
        return self.round_messages(
            case, first, [r['Assistant'] for r in rounds], video_llm=video_llm)

    def build_policy(self, case, messages):
        # No policy LLM: the next turn is a fixed function of the dialogue and the
        # evaluated model's own replies. The checklist judge runs in evaluate().
        case = self._case(case)
        content = messages[0]['content'] if messages else []
        video_llm = any(item.get('type') == 'video' for item in content)
        return MTVideoBenchPolicy(
            self, case, self._first_inference_round(case), video_llm=video_llm)

    # ------------------------------------------------------------------ frames

    def save_video_frames(self, video):
        """Sample the video once (official 128-frame uniform protocol) and reuse the frames."""
        with self._frames_lock:
            paths = self._frames.get(video)
            if paths is not None:
                return paths
            paths = self._sample_frames(video)
            self._frames[video] = paths
            return paths

    def save_video_mp4(self, video):
        """Re-encode the sampled frames as a 15fps mp4 for native-video models."""
        with self._video_lock:
            path = self._videos.get(video)
            if path is not None:
                return path
            path = self._encode_video(video)
            self._videos[video] = path
            return path

    def _video_path(self, video):
        return osp.join(self.frame_root, video, f'mtvb-{self.nframe}f-15fps.mp4')

    def _encode_video(self, video):
        path = self._video_path(video)
        if osp.exists(path) and osp.getsize(path) > 0:
            return path
        frames = self.save_video_frames(video)
        with portalocker.Lock(osp.join(self.frame_root, video + '.mtvb.mp4.lock'),
                              'w', timeout=600):
            if osp.exists(path) and osp.getsize(path) > 0:
                return path
            import cv2
            first = cv2.imread(frames[0])
            height, width = first.shape[:2]
            if max(width, height) > 720:
                scale = 720 / max(width, height)
                width, height = int(round(width * scale)), int(round(height * scale))
            tmp = path + '.tmp.mp4'
            writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))
            try:
                for frame_path in frames:
                    frame = cv2.imread(frame_path)
                    if (frame.shape[1], frame.shape[0]) != (width, height):
                        frame = cv2.resize(frame, (width, height))
                    writer.write(frame)
            finally:
                writer.release()
            os.replace(tmp, path)
        return path

    def _frame_dir(self, video):
        return osp.join(self.frame_root, video, f'mtvb-{self.nframe}f')

    def _sample_frames(self, video):
        frame_dir = self._frame_dir(video)
        meta_file = osp.join(frame_dir, 'meta.json')

        def cached():
            if not osp.exists(meta_file):
                return None
            paths = [osp.join(frame_dir, name) for name in load(meta_file)]
            return paths if all(osp.exists(p) for p in paths) else None

        paths = cached()
        if paths is not None:
            return paths
        os.makedirs(frame_dir, exist_ok=True)
        with portalocker.Lock(osp.join(self.frame_root, video + '.mtvb.lock'), 'w', timeout=600):
            paths = cached()
            if paths is not None:
                return paths
            import decord
            vid_path = self._fetch(f'videos/{video}.mp4')
            vr = decord.VideoReader(vid_path)
            total = len(vr)
            if total > self.nframe:
                indices = np.linspace(0, total - 1, self.nframe, dtype=int).tolist()
            else:
                indices = list(range(total))
            frames = vr.get_batch(indices).asnumpy()
            names = [f'frame-{k}-of-{len(indices)}.jpg' for k in range(1, len(indices) + 1)]
            for name, arr in zip(names, frames):
                Image.fromarray(arr).save(osp.join(frame_dir, name))
            dump(names, meta_file)
            return [osp.join(frame_dir, name) for name in names]

    # -------------------------------------------------------------- evaluation

    @staticmethod
    def _parse_prediction(value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return ast.literal_eval(value)
        return value

    def _judge_once(self, judge_model, model_answer, checklist, attempts=5):
        """One official judge run; returns the parsed dict or None."""
        checklist_str = "\n".join(f"{k}: {v}" for k, v in checklist.items() if k[0] == "Q")
        prompt = JUDGE_PROMPT.format(model_answer=model_answer, checklist_str=checklist_str)
        for _ in range(attempts):
            try:
                reply = judge_model.generate([dict(type='text', value=prompt)])
                if has_failure(reply):
                    continue
                result = json.loads(extract_json(reply))
                if isinstance(result, dict):
                    return result
            except Exception:
                continue
        return None

    def _judge_task(self, judge_model, model_answer, checklist):
        answers = [v for k, v in checklist.items() if k[0] == "A"]
        result = self._judge_once(judge_model, model_answer, checklist)
        if result is None:
            return None
        correct = sum(str(r).lower() == str(a).lower() for r, a in zip(result.values(), answers))
        acc = round(correct / len(result), 2) if len(result) > 0 else 0.0
        return dict(results=result, acc=acc)

    def evaluate(self, eval_file, **judge_kwargs):
        model = judge_kwargs.pop('model', None) or self.DEFAULT_JUDGE_MODEL
        nproc = judge_kwargs.pop('nproc', 4)
        tmp_file = get_intermediate_file_path(eval_file, f'_{model}', 'pkl')
        score_file = get_intermediate_file_path(eval_file, f'_{model}_score', 'csv')
        detail_file = get_intermediate_file_path(eval_file, f'_{model}_detail', 'json')

        data = load(eval_file)
        # unit: (row index, condition, round index) -> dict(answer, checklist, ability)
        units = {}
        rows_by_index = {int(r['index']): r for r in self.data.to_dict('records')}
        for _, line in data.iterrows():
            row = rows_by_index[int(line['index'])]
            preds = self._parse_prediction(line['prediction'])
            if isinstance(preds, dict):
                preds = [preds]
            rounds = self._json(row, 'mtqa')
            checklist_items = self._json(row, 'checklist') or []
            by_round = {item['round index']: item for item in checklist_items}
            for pred in preds:
                responses = pred.get('responses', []) if isinstance(pred, dict) else []
                for k, round_key in enumerate(rounds):
                    round_idx = _round_index(round_key)
                    if round_idx not in by_round:
                        continue
                    item = {k: v for k, v in by_round[round_idx].items()
                            if k.startswith(('Q', 'A'))}
                    pos = round_idx - 1
                    answer = responses[pos] if 0 <= pos < len(responses) else None
                    failed = answer is None or has_failure(answer)
                    units[(int(row['index']), pred['condition'], round_idx)] = dict(
                        ability=row['ability'], event=str(row['event_index']), video=row['video'],
                        answer=answer, checklist=item, failed=failed)

        cache = load(tmp_file) if osp.exists(tmp_file) else {}
        tasks, keys = [], []
        for (idx, cond, rnd), unit in units.items():
            if unit['failed']:
                continue
            for run in range(self.judge_rounds):
                key = f'{idx}|{cond}|{rnd}|{run}'
                if key not in cache:
                    tasks.append((unit['answer'], unit['checklist']))
                    keys.append(key)
        if tasks:
            judge_model = build_judge(model=model, **judge_kwargs)
            track_progress_rich(
                lambda answer, checklist: self._judge_task(judge_model, answer, checklist),
                tasks, nproc=nproc, chunksize=nproc, keys=keys, save=tmp_file)
            cache = load(tmp_file)

        # round -> event -> ability -> overall, per condition (official eval.py aggregation)
        round_acc, details = {}, []
        for (idx, cond, rnd), unit in units.items():
            if unit['failed']:
                # Inference failures stay in the denominator as 0.
                acc, runs = 0.0, []
            else:
                runs = [cache.get(f'{idx}|{cond}|{rnd}|{r}') for r in range(self.judge_rounds)]
                runs = [r for r in runs if r is not None]
                acc = round(sum(r['acc'] for r in runs) / len(runs), 2) if runs else None
            details.append(dict(index=idx, condition=cond, round=rnd, ability=unit['ability'],
                                acc=acc, inference_failed=unit['failed'],
                                valid_judge_runs=len(runs)))
            if acc is not None:
                round_acc.setdefault((cond, unit['ability'], idx), []).append(acc)
        rows = []
        for cond in sorted({c for c, _, _ in round_acc}):
            scores = {}
            counts = {}
            for (c, ability, idx), accs in round_acc.items():
                if c != cond:
                    continue
                scores.setdefault(ability, []).append(round(sum(accs) / len(accs), 2))
                counts[ability] = counts.get(ability, 0) + len(accs)
            ability_means = {}
            for ability in ABILITY_FILES:
                if ability in scores:
                    ability_means[ability] = sum(scores[ability]) / len(scores[ability])
                    rows.append(dict(condition=cond, ability=ability,
                                     mean_mtqa=ability_means[ability],
                                     num_events=len(scores[ability]), num_rounds=counts[ability]))
            if ability_means:
                rows.append(dict(condition=cond, ability='Overall',
                                 mean_mtqa=sum(ability_means.values()) / len(ability_means),
                                 num_events=sum(len(v) for v in scores.values()),
                                 num_rounds=sum(counts.values())))
        result = pd.DataFrame(rows)
        dump(result, score_file)
        dump(details, detail_file)
        return result
