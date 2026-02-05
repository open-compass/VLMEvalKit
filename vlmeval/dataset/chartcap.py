from .image_base import ImageBaseDataset
from ..smp import *
import warnings

class ChartCapDataset(ImageBaseDataset):
    
    TYPE = 'Caption'
    DATASET_URL = {
        'ChartCap': '',
        'ChartCapSmall': '',
        'ChartCapProbe': '',
    }
    
    DATASET_MD5 = {
        'ChartCap': '10a0292079120f748eae81af5a1e19da',
        'ChartCapSmall': '6ba51dd13002af11f1efd182c18b85ff',
        'ChartCapProbe': '31b6dd10b39eedf956f34e3dd010b857',
    }

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.megabench.scoring.sacrebleu_bleu import Bleu
        from bert_score import BERTScorer
        from sentence_transformers import SentenceTransformer
        import evaluate
        
        data = load(eval_file)
        
        # Prepare predictions and references
        # Ensure data is sorted or aligned by index if needed, but usually index is preserved
        # data usually contains 'prediction' and 'answer' columns
        
        predictions = [str(x) for x in data['prediction']]
        references = [str(x) for x in data['answer']]
        
        # Metric 1: sacreBLEU
        # Reference implementation uses corpus_bleu
        # vlmeval wrapper: Bleu.match(pred, gt) returns single score or expected aggregated?
        # Looking at Bleu.match source: it calls corpus_bleu(corr, [resp]).score / 100
        # It seems to be designed for single sample or list of list?
        # Let's use sacrebleu directly for corpus level or the wrapper if it supports batch properly.
        # wrapper `Bleu.match(response, correct_answer)` handles lists. 
        # But it returns scalar 0-1 (divided by 100).
        
        bleu_score = Bleu.match(predictions, references) * 100
        
        # Metric 2: ROUGE-L through 'evaluate'
        try:
            rouge = evaluate.load("rouge")
            rouge_results = rouge.compute(predictions=predictions, references=references)
            rouge_l = rouge_results['rougeL']
        except Exception as e:
            warnings.warn(f"Failed to compute ROUGE: {e}")
            rouge_l = 0.0

        # Metric 3: METEOR through 'evaluate'
        try:
            meteor = evaluate.load("meteor")
            meteor_results = meteor.compute(predictions=predictions, references=references)
            meteor_score = meteor_results['meteor']
        except Exception as e:
            warnings.warn(f"Failed to compute METEOR: {e}")
            meteor_score = 0.0

        # Metric 4: BERTScore
        # Refer to uni_svg.py usage
        try:
            # Device handling
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Using roberta-large-mnli or similar is common, but let's stick to default or what uni_svg uses
            # uni_svg uses "en" (which defaults to roberta-large presumably)
            bert_scorer = BERTScorer(lang="en", rescale_with_baseline=False, device=device)
            P, R, F1 = bert_scorer.score(predictions, references)
            bert_score_val = F1.mean().item()
        except Exception as e:
            warnings.warn(f"Failed to compute BERTScore: {e}")
            bert_score_val = 0.0
            
        results = {
            'BLEU_4': bleu_score,
            'ROUGE_L': rouge_l,
            'METEOR': meteor_score,
            'BERTScore': bert_score_val
        }
        
        # Format as requested: dictionary composed of lists, organized into a pandas.DataFrame 
        # or just a dictionary of scores? 
        # Base class says: "The return value of the function is the calculated accuracy and other metrics, 
        # formatted as a dictionary composed of lists, organized into a pandas.DataFrame."
        # However, for captioning, we usually return a single scalar per metric for the whole dataset.
        # But typical evaluate returns a dict (often converted to DF later). 
        # Let's make it a DF with one row or just the dict if allowed.
        # Looking at ImageCaptionDataset, it returns a dict of scores.
        # Let's return the dict, usually frame_eval handles it.
        
        
        print(f"Evaluation Results for ChartCap:")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
        
        # Create CSV output file
        import pandas as pd
        result_df = pd.DataFrame([results])
        result_file = eval_file.replace(f".{eval_file.split('.')[-1]}", "_acc.csv")
        dump(result_df, result_file)
            
        return results