"""VisChainBench (https://arxiv.org/abs/2512.06759) multi-turn integration.

Protocol source: ``auto_test_vlm.py`` of the official HF dataset
``eyehole/VisChainBench`` (pinned revision below). Three task forms:

* ITMR (image-text multi-turn): system + 2 user turns, then one user turn per
  question with labelled option images; stop at the first wrong answer.
* IOMR (image-only multi-turn): system + initial-condition user turn, then two
  user turns per step (condition image, labelled option images); stop at the
  first wrong answer.
* ICIR (in-context image reasoning): system + context turn + question turn;
  the (single) question is always answered, history is kept across questions.

There is no judge LLM: the answer is parsed with the official regex
``ANSWER:\\s*.*?(\\d+)`` and compared with the gold option(s).
"""
import json
import os
import os.path as osp
import re
import shutil
import uuid
import warnings
import zipfile
from copy import deepcopy

import pandas as pd

from ..interactive_mt import Step, TurnPolicy, has_failure
from ..smp import LMUDataRoot, dump, load
from .image_base import ImageBaseDataset
from .interactive_mt import InteractiveMTMixin

HF_REPO = 'eyehole/VisChainBench'
HF_REVISION = 'a07750e169712b2058574c83f0a5820503d5ea4b'
# (zip file, task type); the extracted top-level folder name is read from the zip.
ZIP_FILES = [
    ('CSTask.zip', 'ITMR'), ('DailyTask.zip', 'ITMR'), ('EngTask.zip', 'ITMR'),
    ('ICCSTask.zip', 'ICIR'), ('ICDailyTask.zip', 'ICIR'), ('ICEngTask.zip', 'ICIR'),
    ('PITaskDaily.zip', 'IOMR'), ('PITaskEng.zip', 'IOMR'),
]
READY_FILE = '.vischainbench_ready'

ITMR_SYSTEM = (
    "You are a assistant in multiturn VQA image selection. Now I will give you a task "
    "discroption, a series of questions.\n"
    "There will be multi images to chosse for each question.The image will be labeled on the "
    "corner.You should answer current question with it's label."
    "You should begin your answer with 'ANSWER:'.")
IOMR_SYSTEM = (
    "You are a assistant in pure image condition task. You will be shown an initial image and a "
    "series of images representing situations and options.\n"
    "For each step, you will see a condition image showing the current situation and multiple "
    "option images labeled 1, 2, 3.\n"
    "Your task is to choose the most appropriate option (1, 2, or 3) for the given condition.\n"
    "Your answer should begin with 'ANSWER:' followed by the option number (1, 2, or 3).")
ICIR_SYSTEM = (
    "I will show you some images that contains some relations,Guess which next image to "
    "choose/to do after the image sequences.\n"
    "The image choices are labeled on the corner(like 0,1,2,3...).Your answer should begin with "
    "ANSWER:(image label)")
ICIR_CONTEXT_PROMPT = 'Here are the context images showing the sequence:'
ICIR_QUESTION_PROMPT = 'Please answer with ANSWER: followed by the number.'

# Label font of the official encode_pil_image. If Pillow cannot find Arial.ttf (it also
# searches the system font folders), the official script uses load_default(), and so do we.
OFFICIAL_FONT = 'C:/Windows/Fonts/Arial.ttf'


def extract_answer_number(text):
    """Official parser (auto_test_vlm.extract_answer_number); None if absent."""
    if not isinstance(text, str):
        return None
    match = re.search(r'ANSWER:\s*.*?(\d+)', text)
    return int(match.group(1)) if match else None


def _text(value):
    return dict(type='text', value=value)


def _image(path):
    return dict(type='image', value=path)


def _user(*content):
    return dict(role='user', content=list(content))


def _resolve(task_dir, name):
    """Return the path of ``name`` inside ``task_dir`` or None.

    Names are used verbatim: the official script opens ``os.path.join(task_dir, name)``
    and silently drops a missing image (the released data has some names with stray
    blanks, whose files therefore never load in the official run either).
    """
    path = osp.join(task_dir, name)
    return path if osp.isfile(path) else None


def _zip_member_name(info):
    """Real name of a zip member.

    The archives were written on a Chinese Windows machine: non-ASCII names are GBK bytes
    without the UTF-8 flag, which ``zipfile`` decodes as cp437. The task JSON files refer
    to the GBK-decoded names (e.g. ``triathlon公告_board.jpg``).
    """
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode('cp437').decode('gbk')
    except UnicodeError:
        return info.filename


def _extract_zip(zf, dst):
    dst = osp.realpath(dst)
    for info in zf.infolist():
        target = osp.realpath(osp.join(dst, _zip_member_name(info)))
        if not target.startswith(dst + os.sep):
            raise ValueError(f'Unsafe path in VisChainBench archive: {info.filename}')
        if info.is_dir():
            os.makedirs(target, exist_ok=True)
            continue
        os.makedirs(osp.dirname(target), exist_ok=True)
        with zf.open(info) as src, open(target, 'wb') as out:
            shutil.copyfileobj(src, out)


def _load_font():
    """Same choice as the official encoder: Arial 60, else Pillow's default font.

    Returns (font, tag); the tag names the cache folder of the labelled images.
    """
    from PIL import ImageFont
    try:
        return ImageFont.truetype(OFFICIAL_FONT, 60), 'arial60'
    except IOError:
        return ImageFont.load_default(), 'default_font'


def _label_cache_dir(img_root):
    # The cached pixels depend on the font, so each font gets its own cache folder.
    return osp.join(img_root, f'_labelled_{_load_font()[1]}')


def labelled_image(src, dst, label):
    """Draw ``label`` (red, top-left corner) like the official encoder; cached at ``dst``."""
    if osp.exists(dst):
        return dst
    from PIL import Image, ImageDraw
    img = Image.open(src)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    ImageDraw.Draw(img).text((10, 10), str(label), font=_load_font()[0], fill=(255, 0, 0))
    os.makedirs(osp.dirname(dst), exist_ok=True)
    tmp = f'{dst}.{uuid.uuid4().hex}.tmp'
    # Lossless cache: the API wrapper JPEG-encodes it once, like the official encoder,
    # instead of compressing the small label twice.
    img.save(tmp, format='PNG')
    os.replace(tmp, dst)  # atomic: concurrent workers may label the same image
    return dst


class VisChainPlan:
    """Messages and gold answers of one task, in official order."""

    def __init__(self, system, prelude, turns, golds, stop_on_wrong):
        self.system = system          # system message
        self.prelude = prelude        # user messages before the first turn
        self.turns = turns            # list[list[user message]] one entry per question
        self.golds = golds            # list[list[int]] accepted answer numbers per question
        self.stop_on_wrong = stop_on_wrong


def _labelled_options(img_root, task_dir, names):
    """Load options like the official code: missing files vanish and labels re-number."""
    found = [p for p in (_resolve(task_dir, n) for n in names) if p is not None]
    out = []
    cache_dir = _label_cache_dir(img_root)
    for i, path in enumerate(found):
        # Cache key keeps the source extension and folder: one task ships the same stem
        # as both .jpg and .png, and a shared key would show the wrong label.
        rel = osp.relpath(path, img_root)
        dst = osp.join(cache_dir, f'{rel}__L{i + 1}.png')
        out.append(_image(labelled_image(path, dst, i + 1)))
    return out


def build_plan(img_root, task_type, subset, task_id, specs):
    task_dir = osp.join(img_root, subset, str(task_id))
    if task_type == 'ITMR':
        data = specs[0]
        prelude = [_user(_text('Please proceed with the questions.'))]
        if 'initial_scene_description' in data:
            prelude.append(
                _user(_text('initial_scene_description: ' + data['initial_scene_description'])))
        turns, golds = [], []
        for key, q in data.items():
            if not key.startswith('Q'):
                continue
            options = _labelled_options(img_root, task_dir, [c['image'] for c in q['choices']])
            turns.append([_user(_text(q['question']), *options)])
            golds.append([i + 1 for i, c in enumerate(q['choices']) if c['correct']])
        return VisChainPlan(ITMR_SYSTEM, prelude, turns, golds, True)

    if task_type == 'IOMR':
        data = specs[0]
        init = _resolve(task_dir, data['initial_condition'])
        if init is None:
            raise FileNotFoundError(
                f'VisChainBench IOMR {subset}/{task_id}: initial condition image missing')
        prelude = [_user(_text('Here is the initial condition:'), _image(init))]
        turns, golds = [], []
        for step in data['steps']:
            cond = _resolve(task_dir, step['condition'])
            names = [o['image'] for o in step['options']]
            options = _labelled_options(img_root, task_dir, names)
            if cond is None or len(options) != len(names):
                continue  # official: the whole step is skipped and not counted
            turns.append([
                _user(_text(f"Step {step['step']}: Current condition :"), _image(cond)),
                _user(_text('Options: '), *options),
            ])
            golds.append([ord(step['correct_option']) - ord('A') + 1])
        return VisChainPlan(IOMR_SYSTEM, prelude, turns, golds, True)

    if task_type == 'ICIR':
        turns, golds = [], []
        for data in specs:
            start = data.get('question_start_label', 0)
            total = 0
            for step in data.get('steps', []):
                for image in step.get('images', []):
                    total = max(total, image.get('image_id', 0))
            answer = next(
                (s['answer_label'] for s in data.get('steps', []) if 'answer_label' in s), None)
            if answer is None:
                continue  # official: "No answer_label found" -> question skipped
            # official ranges: context 1..start-2, options start..max_id-1
            context = [p for p in (
                _resolve(task_dir, f'{i}.jpg') for i in range(1, start - 1)) if p]
            options = _labelled_options(
                img_root, task_dir, [f'{i}.jpg' for i in range(start, total)])
            turns.append([
                _user(_text(ICIR_CONTEXT_PROMPT), *[_image(p) for p in context]),
                _user(_text(ICIR_QUESTION_PROMPT), *options),
            ])
            golds.append([answer - start + 1])
        return VisChainPlan(ICIR_SYSTEM, [], turns, golds, False)

    raise ValueError(f'Unknown VisChainBench task type: {task_type}')


class VisChainPolicy(TurnPolicy):
    """Fixed question sequence; the model's own answers stay in the history."""

    def __init__(self, plan, task_type):
        super().__init__()
        self.plan = plan
        self.task_type = task_type
        self.max_turns = len(plan.turns)
        self.state['responses'] = []

    def on_failure(self, message, history, turn, reason='model'):
        # Keep one slot per attempted turn: the official counts the question
        # before the model call, so a failed call is an attempted, wrong turn.
        self.state['responses'].append(message)
        super().on_failure(message, history, turn, reason)

    def step(self, assistant_response, history):
        turn = len(self.state['responses'])
        self.state['responses'].append(assistant_response)
        wrong = extract_answer_number(assistant_response) not in self.plan.golds[turn]
        if wrong and self.plan.stop_on_wrong:
            return Step(done=True)
        self.next_turn += 1
        if self.next_turn >= self.max_turns:
            return Step(done=True)
        # IOMR / ICIR: both user messages of the next question, kept separate.
        return Step(messages=deepcopy(self.plan.turns[self.next_turn]))

    def finalize(self):
        failure = self.state.get('failure')
        return json.dumps(dict(
            task_type=self.task_type,
            responses=self.state['responses'],
            failure=None if failure is None else dict(
                reason=failure['reason'], turn=failure['turn']),
        ), ensure_ascii=False)


class VisChainBench(InteractiveMTMixin, ImageBaseDataset):

    TYPE = 'VQA'
    DEFAULT_JUDGE_MODEL = None  # rule-based scoring only; no judge LLM
    DEFAULT_POLICY_MODEL = None  # the tested model is the only LLM
    DATASET_URL = {
        'VisChainBench': '', 'VisChainBench_ITMR': '',
        'VisChainBench_ICIR': '', 'VisChainBench_IOMR': '',
    }
    DATASET_MD5 = {}
    TASK_TYPES = ('ITMR', 'ICIR', 'IOMR')

    # ---- data ---------------------------------------------------------------

    def load_data(self, dataset):
        root = LMUDataRoot()
        # All four dataset names share one image folder and one table.
        self.img_root = osp.join(root, 'images', 'VisChainBench')
        self.data_path = osp.join(root, 'VisChainBench.tsv')
        ready = osp.join(self.img_root, READY_FILE)
        if not (osp.exists(self.data_path) and osp.exists(ready)):
            self._prepare_data()
        data = load(self.data_path)
        if dataset != 'VisChainBench':
            data = data[data['task_type'] == dataset.split('_')[-1]].reset_index(drop=True)
        return data

    def _prepare_data(self):
        from huggingface_hub import hf_hub_download
        os.makedirs(self.img_root, exist_ok=True)
        rows = []
        for zip_name, task_type in ZIP_FILES:
            path = hf_hub_download(
                HF_REPO, f'VisChainBench/{zip_name}', repo_type='dataset', revision=HF_REVISION)
            with zipfile.ZipFile(path) as zf:
                subset = _zip_member_name(zf.infolist()[0]).split('/')[0]
                _extract_zip(zf, self.img_root)
            subset_dir = osp.join(self.img_root, subset)
            tasks = sorted(
                os.listdir(subset_dir),
                key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x))
            for task_id in tasks:
                task_dir = osp.join(subset_dir, task_id)
                if not osp.isdir(task_dir):
                    continue
                files = [f for f in os.listdir(task_dir) if f.endswith('.json')]
                if not files:
                    warnings.warn(
                        f'VisChainBench: no JSON in {subset}/{task_id}, '
                        'skipped as in the official script.')
                    continue
                if task_type == 'ICIR':  # official order: numeric file name, task.json last
                    files.sort(key=lambda x: (
                        int(x.split('.')[0]) if x.split('.')[0].isdigit() else float('inf')))
                else:
                    files = files[:1]
                specs = [json.load(open(osp.join(task_dir, f), encoding='utf-8')) for f in files]
                plan = build_plan(self.img_root, task_type, subset, task_id, specs)
                rows.append(dict(
                    task_type=task_type, subset=subset, task_id=task_id,
                    spec=json.dumps(specs, ensure_ascii=False),
                    answer=json.dumps(plan.golds), n_turns=len(plan.golds)))
        data = pd.DataFrame(rows)
        data.insert(0, 'index', range(len(data)))
        dump(data, self.data_path)
        with open(osp.join(self.img_root, READY_FILE), 'w') as f:
            f.write(HF_REVISION)

    # ---- InteractiveMTMixin -------------------------------------------------

    def _plan(self, case):
        if isinstance(case, int):
            case = self.data.iloc[case]
        return build_plan(self.img_root, case['task_type'], case['subset'], case['task_id'],
                          json.loads(case['spec']))

    def build_prompt(self, case):
        """Full first-call context: system + prelude users + the first question's user messages."""
        plan = self._plan(case)
        if not plan.turns:
            raise ValueError(
                f"VisChainBench task {case['subset']}/{case['task_id']} "
                'has no answerable question.')
        return ([dict(role='system', content=[_text(plan.system)])]
                + plan.prelude + deepcopy(plan.turns[0]))

    def build_policy(self, case, messages):
        return VisChainPolicy(self._plan(case), case['task_type'])

    # ---- evaluation ---------------------------------------------------------

    @staticmethod
    def score_task(golds, prediction, stop_on_wrong):
        """Official counting for one task: (attempted, correct, finished)."""
        try:
            record = json.loads(prediction)
            responses = record['responses']
            failed = record.get('failure') is not None
        except (TypeError, ValueError, KeyError):
            responses, failed = [], True
        attempted = correct = 0
        finished = len(golds) > 0
        for turn, gold in enumerate(golds):
            if turn >= len(responses):
                finished = False  # stopped early / never reached
                break
            attempted += 1
            is_failed_call = failed and turn == len(responses) - 1
            ok = (not is_failed_call) and extract_answer_number(responses[turn]) in gold
            if ok:
                correct += 1
            else:
                finished = False
                if stop_on_wrong or is_failed_call:
                    break
        return attempted, correct, finished

    def evaluate(self, eval_file, **judge_kwargs):
        from ..smp import get_intermediate_file_path
        data = load(eval_file)
        assert len(data) == len(self.data), 'The prediction file does not cover the whole dataset.'
        rows = []
        for _, line in data.iterrows():
            stop = line['task_type'] != 'ICIR'
            att, cor, fin = self.score_task(
                json.loads(line['answer']), str(line['prediction']), stop)
            rows.append(dict(
                index=line['index'], task_type=line['task_type'], subset=line['subset'],
                attempted=att, correct=cor, finished=int(fin),
                has_failure=int(has_failure(str(line['prediction'])))))
        detail = pd.DataFrame(rows)
        dump(detail, get_intermediate_file_path(eval_file, '_detail', 'csv'))

        def summarize(frame):
            return dict(CA=100 * frame['correct'].sum() / max(frame['attempted'].sum(), 1),
                        TC=100 * frame['finished'].sum() / max(len(frame), 1),
                        Tasks=len(frame), Questions=int(frame['attempted'].sum()))

        result, cas = {}, []
        for task_type in self.TASK_TYPES:
            frame = detail[detail['task_type'] == task_type]
            if len(frame) == 0:
                continue
            stats = summarize(frame)
            cas.append(stats['CA'])
            for name, value in stats.items():
                result[f'{task_type}_{name}'] = value
        if cas:
            result['Overall'] = sum(cas) / len(cas)  # README: mean of per-form CA
        for subset, frame in detail.groupby('subset'):
            stats = summarize(frame)
            result[f'{subset}_CA'], result[f'{subset}_TC'] = stats['CA'], stats['TC']
        result['Failed_tasks'] = int(detail['has_failure'].sum())
        score = pd.DataFrame([result])
        dump(score, get_intermediate_file_path(eval_file, '_acc', 'csv'))
        return result
