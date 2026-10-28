"""ImgMT evaluation adapted from the official dataset evaluator.

Source: sarkau/ImgMT-Dataset-For-Evaluating-Text-Image-Translation,
eval/imgmt_eval.py, revision f8c7446c445630c3e0f3086a71e4bc632bcb5c2b.
Dataset repository license: CC BY-SA 4.0
https://creativecommons.org/licenses/by-sa/4.0/

Only the TIT evaluation classes are retained. CLI, unused image evaluation
stubs and unrelated imports are omitted; evaluation behavior is preserved.
"""
import json

import numpy as np
from sacrebleu.metrics import BLEU, CHRF
from scipy.optimize import linear_sum_assignment


class BBox:
    """Calculations for bounding boxes."""

    def calculate_intersection_area(block1, block2):
        (x1, y1, w1, h1) = (block1['x'], block1['y'], block1['w'], block1['h'])
        (x2, y2, w2, h2) = (block2['x'], block2['y'], block2['w'], block2['h'])
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        return x_overlap * y_overlap

    def calculate_area(block):
        return block['w'] * block['h']

    def find_max_overlap_reference_block(pip_block, reference_blocks):
        max_overlap_area = 0
        max_overlap_ref_block = None
        for ref_block in reference_blocks:
            overlap_area = BBox.calculate_intersection_area(pip_block, ref_block)
            if overlap_area > max_overlap_area:
                max_overlap_area = overlap_area
                max_overlap_ref_block = ref_block
        return max_overlap_ref_block

    def merge_boxes(indices, pipeline_boxes):
        selected_boxes = [pipeline_boxes[i] for i in indices]
        x = selected_boxes[0]['x']
        y = max((box['y'] for box in selected_boxes))
        h = max((box['h'] for box in selected_boxes))
        last_box = selected_boxes[-1]
        w = abs(last_box['x'] + last_box['w'] - x)
        return {'x': x, 'y': y, 'w': w, 'h': h}


class TextMetrics:
    """CER calculation"""

    def character_error_rate(predicted, reference):
        distances = np.zeros((len(predicted) + 1, len(reference) + 1))
        for t1 in range(len(predicted) + 1):
            distances[t1][0] = t1
        for t2 in range(len(reference) + 1):
            distances[0][t2] = t2
        for t1 in range(1, len(predicted) + 1):
            for t2 in range(1, len(reference) + 1):
                if predicted[t1 - 1] == reference[t2 - 1]:
                    distances[t1][t2] = distances[t1 - 1][t2 - 1]
                else:
                    a = distances[t1][t2 - 1]
                    b = distances[t1 - 1][t2]
                    c = distances[t1 - 1][t2 - 1]
                    distances[t1][t2] = min(a, b, c) + 1
        return distances[len(predicted)][len(reference)]


class PipelineDataLoader:
    """Read gold and predicted structures."""

    def __init__(self, src_lang, tgt_lang):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def load_gold(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data['source_language'] == self.src_lang and data['target_language'] == self.tgt_lang:
            return {
                'src_lang': data['source_language'],
                'tgt_lang': data['target_language'],
                'source_texts': data['source_texts'],
                'source_text_bounding_boxes': data['source_text_bounding_boxes'],
                'target_texts': data['target_texts'],
                'target_text_bounding_boxes': data['target_text_bounding_boxes'],
            }
        elif data['source_language'] == self.tgt_lang and data['target_language'] == self.src_lang:
            return {
                'src_lang': data['target_language'],
                'tgt_lang': data['source_language'],
                'source_texts': data['target_texts'],
                'source_text_bounding_boxes': data['target_text_bounding_boxes'],
                'target_texts': data['source_texts'],
                'target_text_bounding_boxes': data['source_text_bounding_boxes'],
            }
        else:
            raise ValueError(f'Language pair {self.src_lang}-{self.tgt_lang} not supported')

    def load_predicted(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        extracted = {
            'text_bounding_boxes': data.get('source_text_bounding_boxes', []),
            'texts': data.get('source_texts', []),
            'translated_texts': data.get('target_texts', []),
        }
        return extracted


class Evaluator:
    """Evaluation matching calculations between gold data and predictions."""

    def __init__(self, gold_data, pipeline_data):
        self.gold = gold_data
        self.pipe = pipeline_data

    def pair_blocks(self, pip_text_key, gold_text_key, pip_box_key, gold_box_key, merging):
        coords_pairing = {}
        FP = 0
        pip_boxes = self.pipe[pip_box_key]
        gold_boxes = self.gold[gold_box_key]
        for (pip_idx, pip_block) in enumerate(pip_boxes):
            chosen_gold_block = BBox.find_max_overlap_reference_block(pip_block, gold_boxes)
            if chosen_gold_block is not None:
                gold_idx = gold_boxes.index(chosen_gold_block)
                coords_pairing[pip_idx] = gold_idx
            else:
                FP += 1
        values_set = set(coords_pairing.values())
        FN = sum((g not in values_set for g in range(len(gold_boxes))))
        text_pairing = {}
        if merging:
            sorted_coords_pairing = {}
            for (p, g) in coords_pairing.items():
                sorted_coords_pairing.setdefault(g, []).append(p)
            for (g, p) in sorted_coords_pairing.items():
                sortedby_x = sorted(p, key=lambda x: float(pip_boxes[x]['x']))
                text_p = ' '.join([self.pipe[pip_text_key][x] for x in sortedby_x]).strip()
                text_pairing[text_p] = self.gold[gold_text_key][g]
            coords_pairing = sorted_coords_pairing
        else:
            for (p, g) in coords_pairing.items():
                text_pairing[self.pipe[pip_text_key][p]] = self.gold[gold_text_key][g]
        return (text_pairing, coords_pairing, len(text_pairing), FP, FN)

    def pair_blocks_text_only(self, pip_text_key, gold_text_key, metric=None, model=None,
                              threshold=None, use_ordering=True):
        pip_texts = self.pipe[pip_text_key]
        gold_texts = self.gold[gold_text_key]
        if not pip_texts or not gold_texts:
            return ({}, [], 0, len(pip_texts), len(gold_texts))
        cost_matrix = np.zeros((len(pip_texts), len(gold_texts)))
        if metric == 'chrf':
            th = threshold if threshold is not None else 0.3
            chrf_metric = CHRF()
            for (p_idx, p_text) in enumerate(pip_texts):
                for (g_idx, g_text) in enumerate(gold_texts):
                    score = chrf_metric.sentence_score(p_text, [g_text]).score / 100.0
                    if score < th:
                        cost_matrix[p_idx][g_idx] = float('inf')
                    else:
                        cost_matrix[p_idx][g_idx] = 1.0 - score
        elif metric == 'comet' and model:
            chrf_th = 0.2
            comet_th = threshold if threshold is not None else 0.4
            chrf_metric = CHRF()
            samples = []
            index_mapping = []
            gold_sources = self.gold.get('source_texts', [''] * len(gold_texts))
            for (p_idx, p_text) in enumerate(pip_texts):
                for (g_idx, g_text) in enumerate(gold_texts):
                    c_score = chrf_metric.sentence_score(p_text, [g_text]).score / 100.0
                    if c_score < chrf_th:
                        cost_matrix[p_idx][g_idx] = float('inf')
                    else:
                        src_text = gold_sources[g_idx] if g_idx < len(gold_sources) else ''
                        samples.append({'src': src_text, 'mt': p_text, 'ref': g_text})
                        index_mapping.append((p_idx, g_idx))
            if samples:
                model_output = model.predict(samples, batch_size=64, gpus=0, progress_bar=False)
                scores = model_output['scores']
                for (score, (p_idx, g_idx)) in zip(scores, index_mapping):
                    if score < comet_th:
                        cost_matrix[p_idx][g_idx] = float('inf')
                    else:
                        cost_matrix[p_idx][g_idx] = 1.0 - score
            else:
                cost_matrix.fill(float('inf'))
        else:
            th = threshold if threshold is not None else 0.4
            for (p_idx, p_text) in enumerate(pip_texts):
                for (g_idx, g_text) in enumerate(gold_texts):
                    edit_distance = TextMetrics.character_error_rate(p_text, g_text)
                    max_len = max(len(p_text), len(g_text))
                    rel_cer = edit_distance / max_len if max_len > 0 else edit_distance
                    if rel_cer > th:
                        cost_matrix[p_idx][g_idx] = float('inf')
                    else:
                        cost_matrix[p_idx][g_idx] = edit_distance
        if use_ordering:
            len_p = len(pip_texts)
            len_g = len(gold_texts)
            for p_idx in range(len_p):
                for g_idx in range(len_g):
                    if cost_matrix[p_idx][g_idx] != float('inf'):
                        ordering_score = abs(p_idx / len_p - g_idx / len_g)
                        cost_matrix[p_idx][g_idx] += ordering_score
        (pip_ind, gold_ind) = linear_sum_assignment(cost_matrix)
        text_pairing = {}
        paired_gold_indices = []
        TP = 0
        used_pip_indices = set()
        used_gold_indices = set()
        for (p, g) in zip(pip_ind, gold_ind):
            if cost_matrix[p][g] == float('inf') or np.isinf(cost_matrix[p][g]):
                continue
            text_pairing[pip_texts[p]] = gold_texts[g]
            paired_gold_indices.append(g)
            used_pip_indices.add(p)
            used_gold_indices.add(g)
            TP += 1
        FP = len(pip_texts) - TP
        FN = len(gold_texts) - TP
        return (text_pairing, paired_gold_indices, TP, FP, FN)

    def evaluate_translation(self, metric, merging, model=None, threshold=None, use_ordering=True):
        has_boxes = bool(self.pipe.get('text_bounding_boxes'))
        if has_boxes:
            (text_pairing, _, _, _, _) = (
                self.pair_blocks(
                    'translated_texts',
                    'target_texts',
                    'text_bounding_boxes',
                    'target_text_bounding_boxes',
                    merging)
            )
            if metric == 'comet' and model:
                (text_pairing_src, _, _, _, _) = (
                    self.pair_blocks(
                        'translated_texts',
                        'source_texts',
                        'text_bounding_boxes',
                        'target_text_bounding_boxes',
                        merging)
                )
                paired_gold_indices = []
        else:
            (text_pairing, paired_gold_indices, _, _, _) = (
                self.pair_blocks_text_only(
                    'translated_texts',
                    'target_texts',
                    metric=metric,
                    model=model,
                    threshold=threshold,
                    use_ordering=use_ordering)
            )
        hypotheses = list(text_pairing.keys())
        references = list(text_pairing.values())
        if not hypotheses:
            return 0.0
        if metric == 'bleu':
            return round(BLEU().corpus_score(hypotheses, [references]).score, 3)
        elif metric == 'chrf':
            return round(CHRF().corpus_score(hypotheses, [references]).score, 3)
        elif metric == 'comet' and model:
            if has_boxes:
                sources = list(text_pairing_src.values())
            else:
                gold_sources = self.gold.get('source_texts', [])
                sources = [gold_sources[g_idx] for g_idx in paired_gold_indices if g_idx < len(gold_sources)]
            data_comet = [{'src': s, 'mt': h, 'ref': r} for (s, h, r) in zip(sources, hypotheses, references)]
            model_output = model.predict(data_comet, batch_size=8, gpus=0)
            return round(model_output['system_score'], 3)
        return 0.0

    def evaluate_text_detection(self, merging, rendered, threshold=None, use_ordering=True):
        gold_text_key = 'source_texts'
        gold_box_key = 'source_text_bounding_boxes'
        pip_text_key = 'texts'
        pip_box_key = 'text_bounding_boxes'
        if rendered:
            gold_text_key = 'target_texts'
            gold_box_key = 'target_text_bounding_boxes'
            pip_text_key = 'rendered_texts'
            pip_box_key = 'rendered_text_bounding_boxes'
        if not self.pipe[pip_text_key]:
            return (0.0, 0.0, 0.0)
        has_boxes = bool(self.pipe.get(pip_box_key))
        if has_boxes:
            (text_pairing, coords_pairing, TP, FP, FN) = (
                self.pair_blocks(
                    pip_text_key,
                    gold_text_key,
                    pip_box_key,
                    gold_box_key,
                    merging)
            )
            gold_areas = sum([BBox.calculate_area(g) for g in self.gold[gold_box_key]])
            pip_areas = sum([BBox.calculate_area(p) for p in self.pipe[pip_box_key]])
            intersection_areas = 0
            if merging:
                for (g, indices) in coords_pairing.items():
                    merge_box = BBox.merge_boxes(indices, self.pipe[pip_box_key])
                    intersection_areas += BBox.calculate_intersection_area(self.gold[gold_box_key][g], merge_box)
            else:
                intersection_areas = sum([
                    BBox.calculate_intersection_area(p, self.gold[gold_box_key][coords_pairing[i]])
                    for i, p in enumerate(self.pipe[pip_box_key]) if i in coords_pairing
                ])
            precision_BBA = intersection_areas / pip_areas if pip_areas != 0 else 0
            recall_BBA = intersection_areas / gold_areas if gold_areas != 0 else 0
            f1_score_BBA = (
                2 * precision_BBA * recall_BBA / (precision_BBA + recall_BBA) if
                precision_BBA + recall_BBA != 0 else 0
            )
            f1_score_BBA = round(f1_score_BBA, 3)
        else:
            (text_pairing, _, TP, FP, FN) = (
                self.pair_blocks_text_only(
                    pip_text_key,
                    gold_text_key,
                    metric='cer',
                    threshold=threshold,
                    use_ordering=use_ordering)
            )
            f1_score_BBA = None
        precision_BBC = TP / (TP + FP) if TP + FP != 0 else 0
        recall_BBC = TP / (TP + FN) if TP + FN != 0 else 0
        f1_score_BBC = (
            2 * precision_BBC * recall_BBC / (precision_BBC + recall_BBC) if
            precision_BBC + recall_BBC != 0 else 0
        )
        total_cer = 0
        length_refs = sum([len(word) for word in text_pairing.values()])
        for (text_e, text_g) in text_pairing.items():
            total_cer += TextMetrics.character_error_rate(text_e, text_g)
        norm_CER = round(total_cer / length_refs, 3) if length_refs != 0 else round(total_cer, 3)
        return (norm_CER, f1_score_BBA, round(f1_score_BBC, 3))
