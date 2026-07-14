import json

def read_md_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    
    return content 

def save_paired_result(preds, gts, save_path):
    save_result = []
    formula_id = 0
    for gt, pred in zip(gts, preds):
        save_result.append({ 
            "gt": gt,
            "pred": pred,
            "img_id": formula_id
        })
        formula_id += 1
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_result, f, indent=4, ensure_ascii=False)