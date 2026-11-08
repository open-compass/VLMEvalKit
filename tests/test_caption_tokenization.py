import ast
import shutil
from pathlib import Path

import pytest


def load_scorer():
    source = Path(__file__).resolve().parents[1] / 'vlmeval/dataset/image_caption.py'
    cls = next(node for node in ast.parse(source.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == 'COCO_Caption_Scorer')
    namespace = {}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['COCO_Caption_Scorer']


@pytest.mark.skipif(shutil.which('java') is None, reason='COCO PTB tokenization requires Java')
@pytest.mark.parametrize('captions', [
    ['A small red square.', 'A large blue circle.'],
    ["It's a small red square.", "It's a large blue circle."],
])
def test_caption_scores_match_official_ptb_pipeline(captions):
    pytest.importorskip('pycocoevalcap')
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    ground_truth = {str(i): [caption] for i, caption in enumerate(captions)}
    predictions = {str(i): [caption.lower().rstrip('.')] for i, caption in enumerate(captions)}
    original = {key: list(value) for key, value in ground_truth.items()}
    tokenizer = PTBTokenizer()

    def normalize(rows):
        return tokenizer.tokenize({key: [{'caption': value} for value in values]
                                   for key, values in rows.items()})
    reference_gt = normalize(ground_truth)
    reference_predictions = normalize(predictions)
    expected = {}
    for scorer, name in [(Bleu(4), 'Bleu'), (Rouge(), 'ROUGE_L'), (Cider(), 'CIDEr')]:
        score, _ = scorer.compute_score(reference_gt, reference_predictions)
        expected[name] = ([value * 100 for value in score]
                          if isinstance(score, list) else score * 100)
    actual = load_scorer()(predictions, ground_truth).compute_scores()
    for metric, score in expected.items():
        assert actual[metric] == pytest.approx(score)
    assert ground_truth == original
