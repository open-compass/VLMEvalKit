from huggingface_hub import hf_hub_download
import zipfile
import os
import json

from ...smp import *

def create_csv_from_meta(meta_file, object_key, data_dir, out_file):
    with open(meta_file, 'r') as fp:
        meta = json.load(fp)

    data = []
    for entry in meta:
        image_file = entry['image_file']
        image_path = osp.join(f'{data_dir}/real_dataset', image_file)
        image = encode_image_file_to_base64(image_path)
        question = f"Count the exact number of {entry[object_key]} in the image. Assume the pattern of {entry[object_key]} continues behind any black box. Provide the total number of {entry[object_key]} as if the black box were not there. Only count {entry[object_key]} that are visible within the frame (or would be visible without the occluding box). If {entry[object_key]} are partially in the frame (i.e. if any part of {entry[object_key]} are visible), count it. If the {entry[object_key]} would be partially in the frame without the occluding box, count it."
        answer = str(entry["ground_truth"])
        data.append(dict(image=image, question=question, answer=answer, image_file=image_file))
        df = pd.DataFrame(data).sort_values(by='image_file')
        df.to_csv(out_file, index=True, index_label='index')

def create_tsv_real():
    data_root = LMUDataRoot()
    data_dir = osp.join(data_root, 'capture')
    os.makedirs(data_root, exist_ok=True)
    real_zip = hf_hub_download(repo_id="atinp/CAPTURe", filename="real_dataset.zip", repo_type="dataset")
    
    with zipfile.ZipFile(real_zip, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    # rename the extracted folder (originally called dataset) to real_dataset
    os.rename(f"{data_dir}/dataset", f"{data_dir}/real_dataset")

    real_meta = hf_hub_download(repo_id="atinp/CAPTURe", filename="real_metadata.json", repo_type="dataset")
    out_file = os.path.join(data_dir, 'CAPTURE_real.tsv')
    create_csv_from_meta(real_meta, 'object', data_dir, out_file)
    return out_file
    
def create_tsv_synthetic():
    syn_zip  = hf_hub_download(repo_id="atinp/CAPTURe", filename="synthetic_dataset.zip", repo_type="dataset")
    data_root = LMUDataRoot()
    data_dir = osp.join(data_root, 'capture')
    os.makedirs(data_root, exist_ok=True)
    
    with zipfile.ZipFile(syn_zip, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    synth_meta = hf_hub_download(repo_id="atinp/CAPTURe", filename="synthetic_metadata.json", repo_type="dataset")
    out_file = os.path.join(data_dir, 'CAPTURE_synthetic.tsv')
    create_csv_from_meta(synth_meta, 'dot_shape', data_dir, out_file)
    return out_file