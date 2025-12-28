import os
import re
from tqdm import tqdm
import pandas as pd

import json
from os import path as osp
from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import decode_base64_to_image_file, load, dump, get_intermediate_file_path

class TextHaluBench(ImageBaseDataset):
    TYPE = 'TextHaluBench'
    DATASET_URL = {
        'TextHaluBench':'https://huggingface.co/datasets/LinYuanMo/TextHaluBench/resolve/main/TextHaluBench.tsv',
        
    }
    DATASET_MD5 = {
        'TextHaluBench':' aa80286957b252cb92975486abf1d18e '
    }

    
    def build_prompt(self, line):

        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = toliststr(line['image_path']) if self.meta_only else self.dump_image(line)

        question = line['question']

        if isinstance(tgt_path, list):
            msgs = [dict(type='image', value=p) for p in tgt_path]
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        
        return msgs
    def evaluate(self, eval_file, **judge_kwargs):
        
        import re
        import pandas as pd
        import json

        df = load(eval_file) 

        
        total_tp, total_fp, total_fn = 0, 0, 0
        
        total_tp_under, total_fp_under, total_fn_under = 0, 0, 0

        spotting_f1_list = [] 
        under_f1_list = []  


        def clean_spotting_text(text):
            if pd.isna(text):
                return []
            text = str(text)

            text = text.replace("·", "")
            idx = text.find(":")
            if idx != -1:
                text = text[idx + 1:].strip()

            match = re.search(r"\b(is|reads)\b (.+)", text)
            if match:
                text = match.group(2)

            tokens = re.split(r"[ \n]+", text.replace('"', "").replace(".", ""))
            tokens = [t.strip() for t in tokens if t.strip()]
            tokens = list(set([t.upper() for t in tokens if t != "###"]))

            filter_keywords = ["text"]
            unique, seen = [], set()
            for tok in tokens:
                if any(k in tok for k in filter_keywords):
                    continue
                if tok not in seen:
                    unique.append(tok)
                    seen.add(tok)
            return unique


        def get_predicted_letters(response3: str, question: str):
            response3 = response3.replace("·", "").strip()
            question = str(question).replace("·", "").strip()

            matched_letters = []


            matches = re.findall(r"\b([ABCD])\.", response3, flags=re.IGNORECASE)
            if matches:
                matched_letters = [m.upper() for m in matches]
            else:
                options = re.findall(r"([ABCD])\.\s*([^\n]*)", question)
                for letter, content in options:
                    content = content.strip()
                    if content and content in response3:
                        matched_letters.append(letter.upper())


                if not matched_letters:
                    matches = re.findall(r"(?<![A-Za-z0-9])([ABCD])(?![A-Za-z0-9])", response3, flags=re.IGNORECASE)
                    matched_letters = [m.upper() for m in matches]

            return " ".join(sorted(set(matched_letters)))


        def compute_macro_f1(pred_str, ans_str):
            import re
            tokens = re.split(r"[,/ \n]+", str(pred_str).upper())
            pred = [x.strip() for x in tokens if x.strip()]
            tokens = re.split(r"[,/ \n]+", str(ans_str).upper())
            ans = [x.strip() for x in tokens if x.strip()]

            pred_set, ans_set = set(pred), set(ans)
            if not pred_set and not ans_set:
                return 1.0
            if not ans_set:
                return 0.0

            labels = sorted(list(pred_set | ans_set))
            f1_list = []
            for label in labels:
                tp = 1 if label in pred_set and label in ans_set else 0
                fp = 1 if label in pred_set and label not in ans_set else 0
                fn = 1 if label not in pred_set and label in ans_set else 0

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                f1_list.append(f1)

            return sum(f1_list) / len(f1_list)



        for idx, row in df.iterrows():
            gt = row["answer"]
            pred_raw = row["prediction"]
            label = row.get("label")
            question = row.get("question")
            
            if label == "understanding":
                predicted_letters = get_predicted_letters(pred_raw, question)
                
                standardized_pred = ",".join([p.strip() for p in predicted_letters.split() if p.strip()])
                df.at[idx, "prediction"] = standardized_pred

                
                f1 = compute_macro_f1(standardized_pred, gt)
                df.at[idx, "f1"] = f1
                under_f1_list.append(f1)

                

            elif label == "spotting" :
                

                real_groups = clean_spotting_text(gt)
                predicted = clean_spotting_text(pred_raw)
                
                df.at[idx, "prediction"] = " ".join(predicted)

                real_set, pred_set = set(real_groups), set(predicted)
                tp = len(real_set & pred_set)
                fp = len(pred_set - real_set)
                fn = len(real_set - pred_set)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                df.at[idx, "f1"] = f1
                spotting_f1_list.append(f1)
                

        under_macro_f1 = sum(under_f1_list) / len(under_f1_list) if under_f1_list else 0.0
        spotting_macro_f1 = sum(spotting_f1_list) / len(spotting_f1_list) if spotting_f1_list else 0.0

        overall_score = (spotting_macro_f1 + under_macro_f1) / 2  

        details_file = get_intermediate_file_path(eval_file, "_details")
        dump(df, details_file)

        results = {
            "spotting_macro_f1": round(spotting_macro_f1, 4),
            "understanding_macro_f1": round(under_macro_f1, 4),
            "overall_score": round(overall_score, 4),
        }
        result_file = get_intermediate_file_path(eval_file, "_results.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results
