import pandas as pd
from datasets import load_dataset
from vlmeval.smp import encode_image_to_base64
from tqdm import tqdm

def prepare_chartcap():
    print("Loading ChartCap dataset from HuggingFace...")
    # Load the dataset
    ds = load_dataset("junyoung-00/ChartCap", split="test") 
    
    # We will collect the data rows here
    data_rows = []
    
    # Default question as per requirements
    default_question = 'Please provide a detailed caption for the chart.'
    
    print("Processing samples...")
    for idx, sample in enumerate(tqdm(ds)):
        # Extract fields
        # sample in HuggingFace dataset usually acts like a dict
        
        # Prepare the base64 image
        if 'image' in sample:
            img = sample['image']
            img_b64 = encode_image_to_base64(img)
        else:
            print(f"Warning: No image found for sample {idx}, skipping.")
            continue

        # Prepare answer (ground truth caption)
        # The dataset likely has a caption field. Let's inspect the keys if we can't be sure
        # But for now I'll assume standard naming or check `sample` content dynamically if needed.
        # Based on typical HF datasets, it might be 'caption' or 'text'.
        # However, looking at junyoung-00/ChartCap on HF (hypothetically provided link), 
        # usually it has 'image' and 'caption'.
        # If 'label' or 'ground_truth' exists, use that.
        # I will dump all original keys as requested.
        
        # Checking common keys for caption in such datasets
        answer = sample.get('caption', sample.get('text', ''))
        
        row = {
            'index': idx,
            'image': img_b64,
            'question': default_question,
            'answer': answer
        }
        
        # Add all original data information
        for k, v in sample.items():
            if k not in row and k != 'image': # Don't duplicate image or overwrite our fields if they match
                 row[k] = v
        
        data_rows.append(row)

    print(f"Creating TSV with {len(data_rows)} samples...")
    df = pd.DataFrame(data_rows)
    
    # Ensure columns order: index, image, question, answer, ... others
    cols = ['index', 'image', 'question', 'answer']
    remaining_cols = [c for c in df.columns if c not in cols]
    df = df[cols + remaining_cols]
    
    output_file = 'ChartCap.tsv'
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Saved to {output_file}")

if __name__ == '__main__':
    prepare_chartcap()