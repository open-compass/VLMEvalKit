# vlmeval/dataset/uniqa3d.py
# UniQA-3D Benchmark Suite: CLEVR VQA, Relative Camera Pose, and Relative Depth

import multiprocessing as mp
import os
import os.path as osp
import string
import base64
import re
import numpy as np
import pandas as pd
from urllib.parse import urlparse, unquote

from .image_vqa import ImageVQADataset
from .image_mcq import ImageMCQDataset
from .image_base import img_root_map, LMUDataRoot
from ..smp import download_file, load, dump, d2df
from ..smp.file import get_intermediate_file_path
from .utils.vqa_eval import hit_calculate

# ============================================================================
# Shared Helper Functions
# ============================================================================

def _is_url(p: str) -> bool:
    if not isinstance(p, str) or not p:
        return False
    u = urlparse(p)
    return u.scheme in ("http", "https", "s3", "gs", "data", "file")

def _norm_local(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def _to_data_url_from_base64(b64_str: str) -> str:
    """Convert a base64 string to a data URL. Assumes PNG format."""
    return f"data:image/png;base64,{b64_str}"

def _to_data_url(p: str) -> str:
    """Convert a local file path to a data URL."""
    import mimetypes
    mime_type, _ = mimetypes.guess_type(p)
    if mime_type is None:
        mime_type = 'image/png'
    with open(p, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    return f"data:{mime_type};base64,{data}"

def _to_image_part(p: str) -> dict:
    """
    Return an OpenAI-compatible image content part.
    - URLs (http/https/s3/gs/data/file) are passed through.
    - Local filesystem paths are converted to data URLs.
    """
    if not p:
        return {}
    if _is_url(p):
        # pass through as URL (OpenAI supports http/https/data/file)
        return {"type": "image_url", "image_url": {"url": p}}
    p_abs = _norm_local(p)
    return {"type": "image_url", "image_url": {"url": _to_data_url(p_abs)}}

def _has_text(x) -> bool:
    return isinstance(x, str) and x.strip() != ""


# ============================================================================
# CLEVR VQA Evaluation Utilities
# ============================================================================

def _normalize_clevr_answer(answer):
    """
    Normalize answer according to CLEVR VQA evaluation rules.
    
    Args:
        answer: Can be bool, int, or str
        
    Returns:
        Normalized string answer
    """
    if isinstance(answer, bool):
        return "true" if answer else "false"
    elif isinstance(answer, int):
        return str(answer)
    else:
        return str(answer)

def _normalize_clevr_prediction(pred):
    """
    Normalize model prediction according to CLEVR VQA evaluation rules.
    
    Args:
        pred: Model prediction string
        
    Returns:
        Normalized string
    """
    if not isinstance(pred, str):
        pred = str(pred)
    
    # Extract first word/phrase before newline or period (answer should be first)
    first_line = pred.split('\n')[0].strip()
    
    # Try to extract just the first word/phrase before period
    if '.' in first_line:
        first_part = first_line.split('.')[0].strip()
    else:
        first_part = first_line.split()[0].strip() if first_line.split() else first_line.strip()
    
    pred = first_part
    
    # Convert to lowercase
    pred = pred.lower()
    
    # Remove periods and spaces
    pred = pred.replace('.', '').replace(' ', '')
    
    # Replace yes/no with true/false
    pred = pred.replace('yes', 'true')
    pred = pred.replace('no', 'false')
    
    return pred

def _process_clevr_line(line):
    """
    Process a single line for CLEVR VQA evaluation.
    
    Args:
        line: Dictionary with 'answer' and 'prediction' keys
        
    Returns:
        Dictionary with 'gt', 'pred', 'match' keys
    """
    gt = line['answer']
    pred = line['prediction']
    
    # Normalize ground truth
    gt_normalized = _normalize_clevr_answer(gt)
    
    # Normalize prediction
    pred_normalized = _normalize_clevr_prediction(pred)
    
    # Compare
    match_score = 1.0 if gt_normalized == pred_normalized else 0.0
    
    return {
        'gt': gt_normalized,
        'pred': pred_normalized,
        'match': [match_score]
    }

# ============================================================================
# UniQA3D RelPose Helper Functions
# ============================================================================

def _letters_present_in_row(row) -> list:
    cols = []
    for k in string.ascii_uppercase:
        if k in row.index:
            val = row[k]
            if isinstance(val, float) and pd.isna(val):
                continue
            if _has_text(str(val)):
                cols.append(k)
    return cols

def _normalize_relpose_answer(ans: str) -> str:
    if not isinstance(ans, str):
        return ""
    ans = ans.strip().upper()
    return ans[:1] if ans[:1] in "ABCDE" else ""

def _normalize_answer(ans: str) -> str:
    """Alias for _normalize_relpose_answer for compatibility."""
    return _normalize_relpose_answer(ans)

# ============================================================================
# UniQA3D RelDepth Helper Functions
# ============================================================================

def _normalize_reldepth_answer(ans: str) -> str:
    """Normalize answer to '1' or '2' for relative depth."""
    if not isinstance(ans, str):
        return ""
    ans = ans.strip()
    
    # Try to extract first digit (1 or 2) from the answer
    if ans and ans[0] in "12":
        if len(ans) == 1 or not ans[1].isdigit():
            return ans[0]
    
    # Try to find "1" or "2" after common prefixes
    match = re.search(r'\b([12])\b', ans)
    if match:
        return match.group(1)
    
    # Fallback: extract first digit if present anywhere
    for char in ans:
        if char in "12":
            return char
    
    return ""

# ============================================================================
# Dataset Classes
# ============================================================================

class UniQA3DClevrVQA(ImageVQADataset):
    """
    UniQA-3D: CLEVR VQA Benchmark
    
    Visual Question Answering benchmark from the UniQA-3D suite.
    """
    TYPE = 'VQA'
    
    SUPPORTED_DATASETS = ["UniQA3DClevrVQA", "CLEVR_VQA"]  # Backward compatibility
    
    DATASET_URL = {
        'UniQA3DClevrVQA': 'https://huggingface.co/TP03/UniQA3D_Dataset/resolve/main/clevr_vqa_optimized.tsv',
        'CLEVR_VQA': 'https://huggingface.co/TP03/UniQA3D_Dataset/resolve/main/clevr_vqa_optimized.tsv',  # Backward compatibility
    }
    
    DATASET_MD5 = {
        'UniQA3DClevrVQA': None,
        'CLEVR_VQA': None,  # Backward compatibility
    }
    
    @classmethod
    def supported_datasets(cls):
        return cls.SUPPORTED_DATASETS
    
    def post_build(self, dataset):
        """Optionally limit dataset to a custom number of samples via UNIQA_N env var."""
        super().post_build(dataset)
        n_limit = os.environ.get('UNIQA_N', None)
        if n_limit is not None:
            n_limit = int(n_limit)
            original_len = len(self.data)
            if original_len > n_limit:
                self.data = self.data.head(n_limit).copy()
                print(f"Limited dataset to {n_limit} samples for testing")
            else:
                print(f"Processing all {len(self.data)} samples (UNIQA_N={n_limit} but dataset has fewer samples)")
    
    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert msgs[-1]['type'] == 'text'
        # Add instruction to put answer first (like original CLEVR evaluation)
        msgs[-1]['value'] += '\nAnswer the question using a single word or phrase. Put your answer first, then explain if needed.'
        return msgs
    
    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate CLEVR VQA using exact match logic."""
        data = load(eval_file)
        dataset = self.dataset_name
        assert 'answer' in data and 'prediction' in data
        
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        
        lt = len(data)
        pool = mp.Pool(16)
        lines = [data.iloc[i] for i in range(lt)]
        
        # Use CLEVR-specific evaluation
        res = pool.map(_process_clevr_line, lines)
        
        data['eval_gt'] = [r['gt'] for r in res]
        data['eval_pred'] = [r['pred'] for r in res]
        data['eval_match'] = [r['match'] for r in res]
        data['eval_score'] = [np.mean(r['match']) for r in res]
        
        detailed_result_file = get_intermediate_file_path(eval_file, '_results')
        dump(data, detailed_result_file)
        
        hit = hit_calculate(res, dataset)
        ret = dict()
        ret['accuracy'] = hit
        
        result_file = get_intermediate_file_path(eval_file, '_acc')
        dump(d2df(ret), result_file)
        
        return ret


class UniQA3DRelPose(ImageMCQDataset):
    """
    UniQA-3D: Relative Camera Pose (two-image MCQ)

    Expected TSV columns:
      index : int
      image_path : str (path/URL to Image1)
      image_path_2 : str (path/URL to Image2)
      question : str
      answer : str (letter 'A'..'E')
      A, B, C, D, E : option text for MCQ (blank if missing)   [or single 'options' column]
      (optional) split : e.g., 'upright', 'rot180'
      (optional) category : e.g., 'yaw', 'pitch', etc.
    """

    TYPE = "MCQ"

    # Accept TSV either positionally or as data=<path>, and avoid kw mismatch with base
    SUPPORTED_DATASETS = ["UniQA3D_RELPOSE", "relative_camera_pose"]
    
    # Default HuggingFace URL for the dataset
    DEFAULT_TSV_URL = "https://huggingface.co/TP03/UniQA3D_Dataset/resolve/main/relative_camera_pose.tsv"

    @classmethod
    def supported_datasets(cls):
        return cls.SUPPORTED_DATASETS

    def __init__(self, data=None, *args, require_split=None, **kwargs):
        # Check if 'dataset' was passed (from build_dataset when using dataset name)
        dataset_name = kwargs.pop('dataset', None)
        
        if args:
            tsv_path = args[0]
            rest_pos = args[1:]
        else:
            tsv_path = data if data is not None else kwargs.pop("data", None)
            rest_pos = ()
        
        # If no path provided, try default locations or use default URL
        if not tsv_path:
            # If dataset_name was provided (e.g., "UniQA3D_RELPOSE"), use it to find the file
            if dataset_name and dataset_name in self.SUPPORTED_DATASETS:
                # Try default local location first
                default_path = "/Users/tianpu/Desktop/Coding/Princeton/JiaLab/Karhan/relative_camera_pose.tsv"
                if osp.exists(default_path):
                    tsv_path = default_path
                else:
                    # Try in LMU data root
                    data_root = LMUDataRoot()
                    potential_path = osp.join(data_root, f"{dataset_name}.tsv")
                    if osp.exists(potential_path):
                        tsv_path = potential_path
                    else:
                        # Try relative_camera_pose.tsv in data root
                        potential_path = osp.join(data_root, "relative_camera_pose.tsv")
                        if osp.exists(potential_path):
                            tsv_path = potential_path
                        else:
                            # Use default URL
                            tsv_path = self.DEFAULT_TSV_URL
            else:
                # No dataset name, try default local location
                default_path = "/Users/tianpu/Desktop/Coding/Princeton/JiaLab/Karhan/relative_camera_pose.tsv"
                if osp.exists(default_path):
                    tsv_path = default_path
                else:
                    # Use default URL
                    tsv_path = self.DEFAULT_TSV_URL
        
        if not tsv_path:
            raise ValueError("UniQA3DRelPose requires the TSV path as first arg, data=<path>, or dataset=<name> with file in default location.")

        # normalize accidental double extension
        if isinstance(tsv_path, str) and tsv_path.endswith(".tsv.tsv") and os.path.exists(tsv_path[:-4]):
            tsv_path = tsv_path[:-4]

        # call parent with positional dataset arg (tsv_path as file path)
        super().__init__(tsv_path, *rest_pos, **kwargs)

        self.require_split = (require_split or "").strip().lower() or None

        if osp.isfile(tsv_path):
            self.dataset_name = self.SUPPORTED_DATASETS[0]
            root = LMUDataRoot()
            self.img_root = osp.join(root, 'images', img_root_map(self.dataset_name))

        if isinstance(self.data, pd.DataFrame):
            for col in ["image_path", "image_path_2", "question", "answer", "split", "category"]:
                if col in self.data.columns:
                    self.data[col] = self.data[col].astype(str)
            if self.require_split and "split" in self.data.columns:
                self.data = self.data[self.data["split"].str.lower() == self.require_split].reset_index(drop=True)
            
            # Limit to N samples if UNIQA_N is set (for testing)
            # Default to 500 samples (full dataset) if not specified
            import os
            n_limit = os.environ.get("UNIQA_N", None)
            if n_limit is None:
                n_limit = 500  # Default to 500 samples (full dataset size)
            n_limit = int(n_limit)
            if n_limit > 0:
                if len(self.data) > n_limit:
                    self.data = self.data.head(n_limit).reset_index(drop=True)
                    print(f"Limited dataset to {n_limit} samples for testing")
                else:
                    # If dataset is smaller than limit, process all
                    print(f"Processing all {len(self.data)} samples (UNIQA_N={n_limit} but dataset has fewer samples)")

    # Load data: handle local files, URLs, or use parent's load_data
    def load_data(self, dataset):
        ds = str(dataset)

        # Normalize file:// → local path
        if ds.startswith("file://"):
            ds = unquote(urlparse(ds).path)

        # accidental ".tsv.tsv"
        if ds.endswith(".tsv.tsv") and os.path.exists(ds[:-4]):
            ds = ds[:-4]

        ds_exp = os.path.abspath(os.path.expanduser(ds))
        if os.path.exists(ds_exp):
            # Local file exists, load directly
            return pd.read_csv(ds_exp, sep="\t", dtype=str).fillna("")

        # Check if it's a URL (http/https)
        parsed = urlparse(ds)
        if parsed.scheme in ("http", "https"):
            data_root = LMUDataRoot()
            os.makedirs(data_root, exist_ok=True)
            # Use DEFAULT_TSV_FILENAME if defined, otherwise default to "relative_camera_pose.tsv"
            file_name = getattr(self, 'DEFAULT_TSV_FILENAME', 'relative_camera_pose.tsv')
            data_path = osp.join(data_root, file_name)
            
            # Download if not exists
            if not osp.exists(data_path):
                download_file(ds, data_path)
            
            # Load the TSV file using standard load function
            return load(data_path)

        # not local and not URL → let parent handle (download/registry/etc.)
        return super().load_data(dataset)

    def build_prompt(self, line):
        """Two images + MCQ text; return VLMEvalKit format message list."""
        if isinstance(line, int):
            row = self.data.iloc[line]
        else:
            row = line

        img1 = (row.get("image_path") or row.get("image1") or "").strip()
        img2 = (row.get("image_path_2") or row.get("image2") or "").strip()
        q    = (row.get("question") or "").strip()

        letters = _letters_present_in_row(row)
        prompt = f"Question: {q}\n"
        if letters:
            prompt += "Options:\n"
            for k in letters:
                prompt += f"{k}. {str(row[k]).strip()}\n"
            prompt += "Please select the correct option letter.\n"
        else:
            opt = (row.get("options") or "").strip()
            if _has_text(opt):
                prompt += f"Options:\n{opt}\nPlease select the correct option letter.\n"
        prompt += "Answer with the option letter only."

        # Return VLMEvalKit format: dict(type='image', value=path) and dict(type='text', value=prompt)
        msgs = []
        if _has_text(img1):
            # Handle relative paths by joining with img_root if needed
            if not _is_url(img1) and not osp.isabs(img1):
                img1 = osp.join(self.img_root, img1) if hasattr(self, 'img_root') else _norm_local(img1)
            msgs.append(dict(type='image', value=img1))
        if _has_text(img2):
            # Handle relative paths by joining with img_root if needed
            if not _is_url(img2) and not osp.isabs(img2):
                img2 = osp.join(self.img_root, img2) if hasattr(self, 'img_root') else _norm_local(img2)
            msgs.append(dict(type='image', value=img2))
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def get_answer(self, line) -> str:
        row = self.data.iloc[line] if isinstance(line, int) else line
        return _normalize_answer(row.get("answer", ""))

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate relative pose using exact matching (answers are 'A'..'E')."""
        from .utils.multiple_choice import report_acc
        
        data = load(eval_file)
        data['prediction'] = [str(x) for x in data['prediction']]
        if 'answer' not in data.columns:
            raise ValueError(f"Missing 'answer' column in evaluation file. Columns: {list(data.columns)}")
        data['answer'] = [str(x) for x in data['answer']]
        
        # Extract choices from question text and add as columns A, B, C, D, E
        # First try to get from original data if available, otherwise extract from question text
        meta = self.data
        meta_map = {int(row['index']): row for _, row in meta.iterrows()}
        
        # Initialize choice columns
        for letter in 'ABCDE':
            data[letter] = None
        
        # Extract choices from original data or question text
        for idx in data.index:
            row = data.loc[idx]
            data_idx = int(row.get('index', idx))
            
            # Try to get choices from original data first
            if data_idx in meta_map:
                meta_row = meta_map[data_idx]
                letters = _letters_present_in_row(meta_row)
                for letter in letters:
                    data.loc[idx, letter] = meta_row.get(letter, "")
            else:
                # Fallback: try to extract from question text using regex
                question = str(row.get('question', ''))
                # Pattern: "A: text; B: text" or "A. text B. text"
                import re
                pattern = r'([A-E])[:.]\s*([^;B-Z]+?)(?=\s*[B-E][:.]|$)'
                matches = re.findall(pattern, question)
                for letter, text in matches:
                    if letter in 'ABCDE':
                        data.loc[idx, letter] = text.strip()
        
        # Normalize predictions and answers to just letters (A-E)
        data['eval_gt'] = data['answer'].apply(_normalize_answer)
        data['eval_pred'] = data['prediction'].apply(_normalize_answer)
        
        # Exact match comparison
        data['hit'] = [(1.0 if gt == pred else 0.0) for gt, pred in zip(data['eval_gt'], data['eval_pred'])]
        
        # Save detailed results
        detailed_result_file = get_intermediate_file_path(eval_file, '_results')
        dump(data, detailed_result_file)
        
        # Calculate accuracy using report_acc for consistency with other MCQ benchmarks
        acc_df = report_acc(data)
        
        # Convert to percentage and save
        result_file = get_intermediate_file_path(eval_file, '_acc')
        
        # report_acc returns a DataFrame with 'split' column and accuracy columns
        # Convert accuracy values to percentages
        for col in acc_df.columns:
            if col != 'split' and col in acc_df.columns:
                acc_df[col] = acc_df[col] * 100
        
        # Rename 'split' column if it contains 'none'
        if 'split' in acc_df.columns:
            acc_df['split'] = acc_df['split'].replace('none', 'Overall')
        
        dump(acc_df, result_file)
        
        return acc_df

    def dump_image(self, line):
        """Override dump_image to handle base64/data URLs in image_path and image_path_2."""
        os.makedirs(self.img_root, exist_ok=True)
        
        # Convert line to dict if needed
        if isinstance(line, int):
            row = self.data.iloc[line]
            line = row.to_dict() if isinstance(row, pd.Series) else dict(row)
        elif isinstance(line, pd.Series):
            line = line.to_dict()
        elif not isinstance(line, dict):
            # Try to convert to dict
            try:
                line = dict(line)
            except:
                # If it's already a dict-like object, use it directly
                pass
        
        # Get image paths (support both naming conventions during transition)
        img1_path = line.get('image_path') or line.get('image1')
        img2_path = line.get('image_path_2') or line.get('image2')
        
        # Convert None to empty string, and handle pandas NaN
        if img1_path is None or (isinstance(img1_path, float) and pd.isna(img1_path)):
            img1_path = ''
        else:
            img1_path = str(img1_path).strip()
            
        if img2_path is None or (isinstance(img2_path, float) and pd.isna(img2_path)):
            img2_path = ''
        else:
            img2_path = str(img2_path).strip()
        
        if not img1_path or not img2_path:
            raise ValueError("Both image_path and image_path_2 are required")
        
        index = line.get('index', '0')
        tgt_paths = []
        
        from ..smp.vlm import decode_base64_to_image_file, read_ok
        
        # Helper function to process an image path
        def process_image(img_path, suffix):
            if img_path.startswith('data:image/'):
                parts = img_path.split(',', 1)
                if len(parts) == 2 and parts[0].endswith('base64'):
                    base64_str = parts[1]
                else:
                    raise ValueError(f"Invalid data URL format: {img_path[:50]}...")
            elif len(img_path) > 100 and not _is_url(img_path) and not osp.isabs(img_path) and not osp.exists(img_path):
                # Likely base64 string (long string that's not a URL or file path)
                base64_str = img_path
            else:
                # Regular file path or URL
                if not _is_url(img_path) and not osp.isabs(img_path):
                    img_path = osp.join(self.img_root, img_path) if hasattr(self, 'img_root') else _norm_local(img_path)
                return img_path, None
            
            # Decode base64 to file
            tgt_path = osp.join(self.img_root, f"{index}_{suffix}.png")
            if not read_ok(tgt_path):
                decode_base64_to_image_file(base64_str, tgt_path)
            return tgt_path, None
        
        # Process both images
        tgt_path1, _ = process_image(img1_path, '0')
        tgt_paths.append(tgt_path1)
        
        tgt_path2, _ = process_image(img2_path, '1')
        tgt_paths.append(tgt_path2)
        
        return tgt_paths

    def peek(self, k=3):
        out = []
        n = min(k, len(self.data))
        for i in range(n):
            row = self.data.iloc[i]
            out.append({
                "index": row.get("index", i),
                "images": [row.get("image_path", ""), row.get("image_path_2", "")],
                "question": row.get("question", ""),
                "answer": _normalize_answer(row.get("answer", "")),
                "split": (row.get("split") or "").strip() or None,
                "category": (row.get("category") or "").strip() or None,
            })
        return out


class UniQA3DRelDepth(ImageMCQDataset):
    """
    UniQA-3D: Relative Depth Estimation (single-image task)
    
    Expected TSV columns:
      index : int
      image_path : str (path/URL/base64 to Image)
      question : str (optional, defaults to fixed prompt)
      answer : str ('1' or '2' - which marker is farther)
      (optional) split : e.g., 'normal', 'flipud'
    
    This class can be inherited to create variants (e.g., flipped versions) by overriding:
    - DEFAULT_TSV_URL: The URL or local path to the TSV file
    - SUPPORTED_DATASETS: List of dataset names that identify this variant
    - DEFAULT_TSV_FILENAME: Local filename to save the downloaded TSV (default: "relative_depth.tsv")
    """

    TYPE = "MCQ"

    SUPPORTED_DATASETS = ["UniQA3D_DEPTH", "relative_depth"]
    DEFAULT_TSV_URL = "https://drive.google.com/uc?export=download&id=18YreBvBZgwMdmVy3Ein9YSbX8RmqcbKo"
    DEFAULT_TSV_FILENAME = "relative_depth.tsv"

    @classmethod
    def supported_datasets(cls):
        return cls.SUPPORTED_DATASETS

    def __init__(self, data=None, *args, require_split=None, **kwargs):
        dataset_name = kwargs.pop('dataset', None)
        
        if args:
            tsv_path = args[0]
            rest_pos = args[1:]
        else:
            tsv_path = data if data is not None else kwargs.pop("data", None)
            rest_pos = ()
        
        # If no path provided, use default URL
        if not tsv_path:
            tsv_path = self.DEFAULT_TSV_URL
        
        if not tsv_path:
            raise ValueError("UniQA3DRelDepth requires the TSV path as first arg, data=<path>, or dataset=<name>.")

        if isinstance(tsv_path, str) and tsv_path.endswith(".tsv.tsv") and os.path.exists(tsv_path[:-4]):
            tsv_path = tsv_path[:-4]

        super().__init__(tsv_path, *rest_pos, **kwargs)

        self.require_split = (require_split or "").strip().lower() or None

        # Always set dataset_name and img_root
        self.dataset_name = self.SUPPORTED_DATASETS[0]
        root = LMUDataRoot()
        self.img_root = osp.join(root, 'images', img_root_map(self.dataset_name))
        # Ensure img_root directory exists
        os.makedirs(self.img_root, exist_ok=True)

        if isinstance(self.data, pd.DataFrame):
            # Create index column if missing (required by parent class)
            if 'index' not in self.data.columns:
                self.data['index'] = range(len(self.data))
            
            for col in ["image_path", "question", "answer", "split"]:
                if col in self.data.columns:
                    self.data[col] = self.data[col].astype(str)
            if self.require_split and "split" in self.data.columns:
                self.data = self.data[self.data["split"].str.lower() == self.require_split].reset_index(drop=True)
            
            # Limit to N samples if UNIQA_N is set (for testing)
            n_limit = os.environ.get("UNIQA_N", None)
            if n_limit is not None:
                n_limit = int(n_limit)
                if n_limit > 0:
                    original_len = len(self.data)
                    if original_len > n_limit:
                        self.data = self.data.head(n_limit).reset_index(drop=True)
                        print(f"[INFO] Limited UniQA3D_DEPTH dataset to {n_limit} samples (out of {original_len} total)")
                    else:
                        print(f"[INFO] Processing all {len(self.data)} samples (UNIQA_N={n_limit} but dataset has fewer samples)")

    def load_data(self, dataset):
        ds = str(dataset)

        if ds.startswith("file://"):
            ds = unquote(urlparse(ds).path)

        if ds.endswith(".tsv.tsv") and os.path.exists(ds[:-4]):
            ds = ds[:-4]

        ds_exp = os.path.abspath(os.path.expanduser(ds))
        if os.path.exists(ds_exp):
            data = pd.read_csv(ds_exp, sep="\t", dtype=str).fillna("")
        else:
            parsed = urlparse(ds)
            if parsed.scheme in ("http", "https"):
                data_root = LMUDataRoot()
                os.makedirs(data_root, exist_ok=True)
                file_name = "relative_depth.tsv"
                data_path = osp.join(data_root, file_name)
                
                if not osp.exists(data_path):
                    download_file(ds, data_path)
                
                data = load(data_path)
            else:
                data = super().load_data(dataset)
        
        # Create index column if missing (required by parent class)
        if 'index' not in data.columns:
            data['index'] = range(len(data))
        
        return data

    def build_prompt(self, line):
        """Single image + fixed prompt for relative depth; return VLMEvalKit format message list."""
        if isinstance(line, int):
            row = self.data.iloc[line]
        else:
            row = line

        img = (row.get("image_path") or "").strip()
        
        question = (row.get("question") or "").strip()
        if not question:
            question = "There are two markers in the image. Which point is farther away from the camera? First answer either 1 or 2, with no additional content, and then explain your answer in separate sentences."
        
        prompt = f"Question: {question}\nAnswer with either 1 or 2, indicating which marker is farther away from the camera."

        msgs = []
        if _has_text(img):
            if img.startswith("data:"):
                pass
            elif len(img) > 100 and not _is_url(img) and not osp.isabs(img) and not osp.exists(img):
                img = _to_data_url_from_base64(img)
            elif not _is_url(img) and not osp.isabs(img):
                img = osp.join(self.img_root, img) if hasattr(self, 'img_root') else _norm_local(img)
            msgs.append(dict(type='image', value=img))
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def get_answer(self, line) -> str:
        row = self.data.iloc[line] if isinstance(line, int) else line
        return _normalize_reldepth_answer(row.get("answer", ""))

    def dump_image(self, line):
        """Override dump_image to handle base64/data URLs in image_path."""
        os.makedirs(self.img_root, exist_ok=True)
        
        if isinstance(line, int):
            line = self.data.iloc[line]
        elif isinstance(line, pd.Series):
            line = line.to_dict()
        
        img_path = line.get('image_path', '')
        if not img_path:
            raise ValueError("image_path is required")
        
        if img_path.startswith('data:image/'):
            parts = img_path.split(',', 1)
            if len(parts) == 2 and parts[0].endswith('base64'):
                base64_str = parts[1]
            else:
                raise ValueError(f"Invalid data URL format: {img_path[:50]}...")
        else:
            base64_str = img_path
        
        index = line.get('index', '0')
        tgt_path = osp.join(self.img_root, f"{index}.png")
        
        from vlmeval.smp.vlm import decode_base64_to_image_file, read_ok
        if not read_ok(tgt_path):
            decode_base64_to_image_file(base64_str, tgt_path)
        
        return [tgt_path]

    def peek(self, k=3):
        out = []
        n = min(k, len(self.data))
        for i in range(n):
            row = self.data.iloc[i]
            out.append({
                "index": row.get("index", i),
                "image": row.get("image_path", ""),
                "question": row.get("question", "") or "There are two markers in the image. Which point is farther away from the camera? First answer either 1 or 2, with no additional content, and then explain your answer in separate sentences.",
                "answer": _normalize_reldepth_answer(row.get("answer", "")),
                "split": (row.get("split") or "").strip() or None,
            })
        return out

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate relative depth using exact matching (answers are '1' or '2')."""
        data = load(eval_file)
        data['prediction'] = [str(x) for x in data['prediction']]
        if 'answer' not in data.columns:
            raise ValueError(f"Missing 'answer' column in evaluation file. Columns: {list(data.columns)}")
        data['answer'] = [str(x) for x in data['answer']]
        
        # Normalize predictions and answers to '1' or '2'
        data['eval_gt'] = data['answer'].apply(_normalize_reldepth_answer)
        data['eval_pred'] = data['prediction'].apply(_normalize_reldepth_answer)
        
        # Exact match comparison
        data['eval_match'] = [(1.0 if gt == pred else 0.0) for gt, pred in zip(data['eval_gt'], data['eval_pred'])]
        data['eval_score'] = data['eval_match']
        
        # Save detailed results
        detailed_result_file = get_intermediate_file_path(eval_file, '_results')
        dump(data, detailed_result_file)
        
        # Calculate accuracy (convert to percentage for consistency with other benchmarks)
        accuracy = data['eval_score'].mean()
        
        # Save accuracy file
        result_file = get_intermediate_file_path(eval_file, '_acc')
        ret = {}
        
        # Save split-wise accuracy if split column exists
        if 'split' in data.columns:
            splits = set(data['split'])
            for sp in splits:
                sub_data = data[data['split'] == sp]
                split_acc = sub_data['eval_score'].mean()
                ret[sp] = split_acc * 100
            ret['Overall'] = accuracy * 100
        else:
            ret['Overall'] = accuracy * 100
        
        ret = d2df(ret)
        dump(ret, result_file)
        
        return ret


# ============================================================================
# Flipped Variants (Inheritable Examples)
# ============================================================================

class UniQA3DRelPoseFlipped(UniQA3DRelPose):
    """
    Flipped version of UniQA-3D Relative Camera Pose (rot180).
    """
    SUPPORTED_DATASETS = ["UniQA3D_RELPOSE_FLIPPED", "relative_camera_pose_flipped"]
    DEFAULT_TSV_URL = "https://huggingface.co/TP03/UniQA3D_Dataset/resolve/main/relative_camera_pose_rot180.tsv"
    DEFAULT_TSV_FILENAME = "relative_camera_pose_flipped.tsv"


class UniQA3DRelDepthFlipped(UniQA3DRelDepth):
    """
    Flipped version of UniQA-3D Relative Depth.
    
    TODO: Update DEFAULT_TSV_URL with the actual flipped dataset URL when available.
    """
    SUPPORTED_DATASETS = ["UniQA3D_DEPTH_FLIPPED", "relative_depth_flipped"]
    DEFAULT_TSV_URL = "https://drive.google.com/uc?export=download&id=1Bu29zfcK-VoIcDC0xi6HvJUkMHcqn5iS"
    DEFAULT_TSV_FILENAME = "relative_depth_flipped.tsv"
    
    def __init__(self, *args, **kwargs):
        if self.DEFAULT_TSV_URL == "TODO_REPLACE_WITH_FLIPPED_RELDEPTH_URL":
            raise ValueError(
                "UniQA3DRelDepthFlipped requires DEFAULT_TSV_URL to be set. "
                "Please update DEFAULT_TSV_URL in uniqa3d.py with the actual flipped dataset URL, "
                "or pass data=<path> when creating the dataset."
            )
        super().__init__(*args, **kwargs)