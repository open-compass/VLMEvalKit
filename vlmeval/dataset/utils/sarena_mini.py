import os
import json
import hashlib
import zipfile
import shutil
import re
import multiprocessing
from collections import defaultdict

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
from PIL import Image
from vlmeval.smp.file import LMUDataRoot

# Configure parallelism before importing tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import metrics
from .SArena.metrics import MetricsConfig, InternSVGMetrics

try:
    from ...smp import load, dump
except ImportError:
    # Fallback to pandas if smp is not available
    def load(file_path):
        if file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        if file_path.endswith('.pkl'):
            return pd.read_pickle(file_path)
        return pd.read_csv(file_path)

    def dump(data, file_path):
        pass


# ================= Configuration =================

SARENA_ROOT = os.path.join(LMUDataRoot(), "SArena_MINI_SrcData")
SARENA_URL = "https://huggingface.co/datasets/JoeLeelyf/SArena-VLMEvalKit/resolve/main/SArena_MINI.zip"
SARENA_ZIP_MD5 = "d49fe7241d16b54aa33a09e90d95af96"
TOKENIZER_PATH = "OpenGVLab/InternVL3-8B"

# ================= Task Definitions =================
TASK_CONFIGS = [
    ("SArena-Icon", "Understanding", "Icon/understanding/sarena_un.jsonl", False),
    ("SArena-Icon", "Generation-T2SVG", "Icon/generation/text2svg.jsonl", False),
    ("SArena-Icon", "Generation-I2SVG", "Icon/generation/img2svg.jsonl", True),
    ("SArena-Icon", "Edit-Color-Complex", "Icon/edit/color_complex.jsonl", False),
    ("SArena-Icon", "Edit-Color-Simple", "Icon/edit/color_simple.jsonl", False),
    ("SArena-Icon", "Edit-Crop", "Icon/edit/crop.jsonl", False),
    ("SArena-Icon", "Edit-Flip", "Icon/edit/flip.jsonl", False),
    ("SArena-Icon", "Edit-Opacity", "Icon/edit/opacity.jsonl", False),
    ("SArena-Icon", "Edit-Outline", "Icon/edit/outline.jsonl", False),
    ("SArena-Icon", "Edit-Rotate", "Icon/edit/rotate.jsonl", False),
    ("SArena-Icon", "Edit-Scale", "Icon/edit/scale.jsonl", False),
    ("SArena-Icon", "Edit-StyleTransform", "Icon/edit/styletransform_openmoji.jsonl", False),
    ("SArena-Icon", "Edit-Translate", "Icon/edit/translate.jsonl", False),
    ("SArena-Illustration", "Generation-T2SVG", "illustration/text2svg.jsonl", False),
    ("SArena-Illustration", "Generation-I2SVG", "illustration/img2svg.jsonl", True),
    ("SArena-Chemistry", "Generation-T2SVG", "chemistry/text2svg.jsonl", False),
    ("SArena-Chemistry", "Generation-I2SVG", "chemistry/img2svg.jsonl", True),
]

EDIT_TASK_MAP = {
    'Edit-Color-Complex': 'color_complex',
    'Edit-Color-Simple': 'color_simple',
    'Edit-Crop': 'crop',
    'Edit-Flip': 'flip',
    'Edit-Rotate': 'rotate',
    'Edit-Scale': 'scale',
    'Edit-Translate': 'translate',
    'Edit-Opacity': 'opacity',
    'Edit-StyleTransform': 'styletransform_openmoji',
    'Edit-Outline': 'outline',
}


# ================= Data Prep Utilities =================

def check_md5(file_path, expected_md5):
    if expected_md5 is None:
        return True
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    print(f"MD5 Check: {hash_md5.hexdigest()} (expected: {expected_md5})")
    return hash_md5.hexdigest() == expected_md5


def download_and_extract_sarena():
    zip_path = os.path.join(SARENA_ROOT, "SArena_MINI.zip")
    if not os.path.exists(SARENA_ROOT):
        os.makedirs(SARENA_ROOT)

    need_download = not os.path.exists(zip_path)
    if not need_download and SARENA_ZIP_MD5:
        if not check_md5(zip_path, SARENA_ZIP_MD5):
            need_download = True

    if need_download:
        print(f"Downloading SArena data from {SARENA_URL}...")
        try:
            response = requests.get(SARENA_URL, stream=True)
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            print(f"Download failed: {e}")
            return

    if not os.path.exists(os.path.join(SARENA_ROOT, "Icon")):
        print("Extracting SArena.zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(SARENA_ROOT)


def load_gt_captions(caption_path):
    caption_map = {}
    if not os.path.exists(caption_path):
        return caption_map

    print(f"  Loading captions from {caption_path}...")
    try:
        with open(caption_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    image_name = item.get('image', '')
                    if not image_name:
                        continue

                    uid_str = image_name.split('.')[0]
                    if uid_str.isdigit():
                        uid = int(uid_str)

                        captions_list = item.get('caption', [])
                        if isinstance(captions_list, list) and len(captions_list) > 0:
                            text = captions_list[0].get('generated_text', '').strip()
                            caption_map[uid] = text
                except Exception:
                    continue
    except Exception as e:
        print(f"  Error reading caption file: {e}")

    return caption_map


# ================= Rendering Utilities =================

def extract_svg_code(prediction_text):
    if not isinstance(prediction_text, str):
        return ""
    pattern = r"```svg(.*?)```"
    matches = re.findall(pattern, prediction_text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    start = prediction_text.find("<svg")
    end = prediction_text.rfind("</svg>")
    if start != -1 and end != -1:
        return prediction_text[start:end + 6]

    if "<?xml" in prediction_text and "</svg>" in prediction_text:
        start = prediction_text.find("<?xml")
        end = prediction_text.rfind("</svg>")
        return prediction_text[start:end + 6]

    return prediction_text.strip()


def _raster_worker_safe(args):
    """
    Worker function compatible with multiprocessing.Pool.
    Includes SIGNAL ALARM to break C-level deadlocks.
    """
    svg_path, output_path, width, height = args
    import os
    import cairosvg
    import xml.etree.ElementTree as ET
    from PIL import Image
    import signal

    class TimeoutException(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutException("Rendering timed out")

    if not os.path.exists(svg_path) or os.path.getsize(svg_path) == 0:
        Image.new('RGB', (width, height), (0, 0, 0)).save(output_path)
        return

    try:
        ET.parse(svg_path)
    except ET.ParseError:
        with open(output_path + ".err", "w") as f:
            f.write(f"Invalid SVG XML: {svg_path}")
        Image.new('RGB', (width, height), (0, 0, 0)).save(output_path)
        return

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)

    try:
        cairosvg.svg2png(url=svg_path,
                         write_to=output_path,
                         background_color='white',
                         output_width=width,
                         output_height=height)
    except TimeoutException:
        with open(output_path + ".err", "w") as f:
            f.write("Timeout: Rendering took longer than 10s")
        Image.new('RGB', (width, height), (0, 0, 0)).save(output_path)
    except Exception as e:
        with open(output_path + ".err", "w") as f:
            f.write(str(e))
        Image.new('RGB', (width, height), (0, 0, 0)).save(output_path)
    finally:
        signal.alarm(0)


# ================= Metrics Processors =================

def evaluate_understanding(gt_file, predictions_map):
    total = 0
    correct = 0
    subject_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    with open(gt_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            cid = item['id']
            subject = item.get('Subject', 'Unknown_Subject')
            gt_ans = item['conversations'][1]['value'].strip()[0]

            # lookup via ID
            pred = predictions_map.get(cid, "")
            if not pred and isinstance(cid, int):
                pred = predictions_map.get(str(cid), "")

            is_correct = False
            pred_choice = re.search(r'[A-D]', pred.strip())
            if pred_choice and pred_choice.group(0) == gt_ans:
                is_correct = True
                correct += 1

            total += 1
            subject_stats[subject]['total'] += 1
            if is_correct:
                subject_stats[subject]['correct'] += 1

    overall_acc = correct / total if total > 0 else 0
    per_subject_results = {}
    for subj, stats in subject_stats.items():
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        per_subject_results[subj] = acc

    return overall_acc, per_subject_results


def evaluate_generation(bench_name, task_type, gt_dir, task_temp_dir, tokenizer_path=None):
    gt_img_dir = os.path.join(gt_dir, 'images')
    gt_svg_dir = os.path.join(gt_dir, 'svg')

    if task_type == 'text2svg':
        config = MetricsConfig(
            use_FID=True,
            use_FID_C=True,
            use_CLIP_Score_T2I=True,
            use_CLIP_Score_I2I=True,
            use_token_length=True
        )
    else:
        config = MetricsConfig(
            use_DINO_Score=True,
            use_LPIPS=True,
            use_SSIM=True,
            use_PSNR=True,
            use_token_length=True
        )

    calculator = InternSVGMetrics(config, tokenizer_path)

    temp_jsonl = os.path.join(task_temp_dir, 'test.jsonl')
    gt_imgs, pred_imgs, gt_svgs, pred_svgs, captions = [], [], [], [], []

    with open(temp_jsonl, 'r') as f:
        for line in f:
            data = json.loads(line)
            uid = data['id']
            p_img = os.path.join(task_temp_dir, 'images', f"{uid}.png")
            p_svg = os.path.join(task_temp_dir, 'svg', f"{uid}.svg")
            g_img = os.path.join(gt_img_dir, f"{uid}.png")
            g_svg = os.path.join(gt_svg_dir, f"{uid}.svg")

            assert os.path.exists(p_img) or os.path.exists(g_img), f"Missing image for ID {uid}"
            pred_imgs.append(
                Image.open(p_img).copy() if os.path.exists(p_img) else Image.new('RGB', (448, 448))
            )
            gt_imgs.append(
                Image.open(g_img).copy() if os.path.exists(g_img) else Image.new('RGB', (448, 448))
            )

            if os.path.exists(p_svg):
                with open(p_svg, 'r') as sf:
                    pred_svgs.append(sf.read())
            else:
                pred_svgs.append("")

            if os.path.exists(g_svg):
                with open(g_svg, 'r') as sf:
                    gt_svgs.append(sf.read())
            else:
                gt_svgs.append("")

            # Uses caption matched in previous logic
            captions.append(data.get('caption', ""))

    batch = {
        'gt_im': gt_imgs,
        'pred_im': pred_imgs,
        'gt_svg': gt_svgs,
        'pred_svg': pred_svgs,
        'caption': captions
    }
    return calculator.calculate_metrics(batch)


def process_editing_batch(gt_root, task_temp_dir, task_name, calculator):
    gt_img_dir = os.path.join(gt_root, 'data', task_name, 'images')
    gt_svg_dir = os.path.join(gt_root, 'data', task_name, 'svg')
    temp_jsonl = os.path.join(task_temp_dir, 'test.jsonl')

    gt_imgs, pred_imgs, gt_svgs, pred_svgs = [], [], [], []
    with open(temp_jsonl, 'r') as f:
        for line in f:
            data = json.loads(line)
            uid = data['id']
            p_img = os.path.join(task_temp_dir, 'images', f"{uid}.png")
            p_svg = os.path.join(task_temp_dir, 'svg', f"{uid}.svg")
            g_img = os.path.join(gt_img_dir, f"{uid}.png")
            g_svg = os.path.join(gt_svg_dir, f"{uid}.svg")

            pred_imgs.append(
                Image.open(p_img).copy() if os.path.exists(p_img) else Image.new('RGB', (448, 448))
            )
            gt_imgs.append(
                Image.open(g_img).copy() if os.path.exists(g_img) else Image.new('RGB', (448, 448))
            )

            if os.path.exists(p_svg):
                with open(p_svg, 'r') as sf:
                    pred_svgs.append(sf.read())
            else:
                pred_svgs.append("")
            if os.path.exists(g_svg):
                with open(g_svg, 'r') as sf:
                    gt_svgs.append(sf.read())
            else:
                gt_svgs.append("")

    batch = {'gt_im': gt_imgs, 'pred_im': pred_imgs, 'gt_svg': gt_svgs, 'pred_svg': pred_svgs}

    current_metrics = calculator.calculate_metrics(batch)
    return current_metrics


# ================= Helper: Load Predictions =================

def load_all_predictions_df(eval_file):
    print(f"Loading predictions from {eval_file}...")
    if eval_file.endswith('.xlsx'):
        df = pd.read_excel(eval_file)
    elif eval_file.endswith('.pkl'):
        df = pd.read_pickle(eval_file)
    elif eval_file.endswith('.csv'):
        df = pd.read_csv(eval_file)
    else:
        df = load(eval_file)
        if isinstance(df, list):
            df = pd.DataFrame(df)

    df.columns = [c.strip() for c in df.columns]

    if 'prediction' not in df.columns:
        print("Warning: 'prediction' column not found. Trying to infer...")
        for col in ['answer', 'output', 'response']:
            if col in df.columns:
                df.rename(columns={col: 'prediction'}, inplace=True)
                break

    if 'l2-category' not in df.columns and 'category' in df.columns:
        print("Warning: 'l2-category' column not found, duplicating 'category' as 'l2-category'...")
        df['l2-category'] = df['category']

    return df


def get_task_predictions_map(df, l1_category, l2_prefix):
    mask_l2 = df['l2-category'].astype(str).str.contains(l2_prefix, case=False, regex=False, na=False)

    possible_l1_cols = ['l2-category']
    if 'category' in df.columns:
        possible_l1_cols.append('category')
    if 'dataset' in df.columns:
        possible_l1_cols.append('dataset')

    mask_l1 = pd.Series(False, index=df.index)
    for col in possible_l1_cols:
        mask_l1 |= df[col].astype(str).str.contains(l1_category, case=False, regex=False, na=False)

    subset = df[mask_l2 & mask_l1].copy()

    print(f"  Filtering for L1='{l1_category}' & L2='{l2_prefix}' -> Found {len(subset)} rows.")

    pred_map = {}

    for _, row in subset.iterrows():
        l2_val = str(row['l2-category'])
        prediction = row['prediction']

        try:
            if '_' in l2_val:
                local_id_str = l2_val.split('_')[-1]
                if local_id_str.isdigit():
                    pred_map[int(local_id_str)] = prediction
                    continue

            if 'id' in row and str(row['id']).isdigit():
                pred_map[int(row['id'])] = prediction
            elif 'index' in row and str(row['index']).isdigit():
                pred_map[int(row['index'])] = prediction

        except Exception:
            pass

    return pred_map


def convert_np_to_native(obj):
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


# ================= Main Function =================

def evaluate_sarena_mini(eval_file, tokenizer_path=TOKENIZER_PATH):
    download_and_extract_sarena()

    all_preds_df = load_all_predictions_df(eval_file)

    eval_dir = os.path.dirname(os.path.abspath(eval_file))
    temp_root = os.path.join(eval_dir, "sarena_mini_svg_temp")
    if not os.path.exists(temp_root):
        os.makedirs(temp_root)

    print(f"Render output directory: {temp_root}")

    final_json_results = defaultdict(lambda: defaultdict(dict))

    # Initialize Metrics Calculator
    edit_config = MetricsConfig(
        use_FID=False, use_FID_C=False,
        use_CLIP_Score_T2I=False, use_CLIP_Score_I2I=False,
        use_DINO_Score=True, use_LPIPS=True, use_SSIM=True, use_PSNR=True,
        use_token_length=True
    )
    edit_calculator = InternSVGMetrics(edit_config, tokenizer_path)
    has_processed_edit = False

    try:
        for category, l2_prefix, rel_path, _ in TASK_CONFIGS:
            src_path = os.path.join(SARENA_ROOT, rel_path)

            if not os.path.exists(src_path):
                print(f"[Warning] File not found: {rel_path}. Skipping task.")
                continue

            task_key = f"{category}_{l2_prefix}"
            print(f"Processing Task: {task_key}")

            current_task_preds = get_task_predictions_map(all_preds_df, category, l2_prefix)

            # ================= [NEW] Determine and Load Caption Map =================
            caption_map = {}
            target_caption_file = None

            if "Generation" in l2_prefix:
                if "Icon" in category:
                    target_caption_file = os.path.join(SARENA_ROOT, "Icon/generation/caption.jsonl")
                elif "Illustration" in category:
                    target_caption_file = os.path.join(SARENA_ROOT, "illustration/caption.jsonl")
                elif "Chemistry" in category:
                    target_caption_file = os.path.join(SARENA_ROOT, "chemistry/caption.jsonl")

            if target_caption_file and os.path.exists(target_caption_file):
                caption_map = load_gt_captions(target_caption_file)
            # ========================================================================

            task_dir = os.path.join(temp_root, task_key.replace(" ", "_"))
            if os.path.exists(task_dir):
                shutil.rmtree(task_dir)

            os.makedirs(os.path.join(task_dir, 'svg'), exist_ok=True)

            is_anim = "Animation" in category
            if is_anim:
                os.makedirs(os.path.join(task_dir, 'video'), exist_ok=True)
                os.makedirs(os.path.join(task_dir, 'video_128'), exist_ok=True)
            else:
                os.makedirs(os.path.join(task_dir, 'images'), exist_ok=True)

            with open(src_path, 'r', encoding='utf-8') as f:
                src_lines = f.readlines()

            render_queue = []
            predictions_map_un = {}
            task_records = []

            match_count = 0

            for line in src_lines:
                src_item = json.loads(line)
                orig_id = src_item['id']

                prediction = current_task_preds.get(orig_id, "")
                if prediction:
                    match_count += 1

                if "Understanding" in l2_prefix:
                    predictions_map_un[orig_id] = str(prediction)
                else:
                    svg_code = extract_svg_code(str(prediction))
                    svg_path = os.path.join(task_dir, 'svg', f"{orig_id}.svg")
                    with open(svg_path, 'w', encoding='utf-8') as sf:
                        sf.write(svg_code)

                    if is_anim:
                        vid_path = os.path.join(task_dir, 'video', f"{orig_id}.mp4")
                        vid_path_128 = os.path.join(task_dir, 'video_128', f"{orig_id}.mp4")
                        render_queue.append({
                            'svg_path': svg_path,
                            'vid_path': vid_path,
                            'vid_path_128': vid_path_128,
                            'type': 'video'
                        })
                    else:
                        img_path = os.path.join(task_dir, 'images', f"{orig_id}.png")
                        render_queue.append({
                            'svg_path': svg_path,
                            'img_path': img_path,
                            'type': 'image'
                        })

                meta = {'id': orig_id}

                gt_caption = caption_map.get(orig_id, "")
                meta['caption'] = gt_caption
                # =======================================================================

                task_records.append(meta)

            print(f"  Matched {match_count}/{len(src_lines)} samples via Local ID.")

            with open(os.path.join(task_dir, 'test.jsonl'), 'w') as tf:
                for r in task_records:
                    tf.write(json.dumps(r) + '\n')

            # --- Metrics Computation & JSON Storage ---

            # 1. Understanding
            if "Understanding" in l2_prefix:
                acc, sub_acc = evaluate_understanding(src_path, predictions_map_un)
                final_json_results[category]["Understanding"]["Overall"] = acc
                for subj, val in sub_acc.items():
                    final_json_results[category]["Understanding"][subj] = val
                print(f"  Accuracy: {acc:.2%}")
                continue

            # --- Rendering ---
            img_queue = [x for x in render_queue if x['type'] == 'image']
            if img_queue:
                print(f"  Rasterizing {len(img_queue)} SVGs using Safe Process Pool...")

                pool_args = [(item['svg_path'], item['img_path'], 448, 448) for item in img_queue]
                num_workers = 32
                ctx = multiprocessing.get_context()

                with ctx.Pool(processes=num_workers, maxtasksperchild=50) as pool:
                    for _ in tqdm(pool.imap_unordered(_raster_worker_safe, pool_args),
                                  total=len(pool_args), desc="Rasterizing"):
                        pass

            # 2. Generation & Editing
            if edit_calculator:
                try:
                    if "Icon" in category and "Edit" in l2_prefix:
                        task_name = EDIT_TASK_MAP.get(l2_prefix, 'unknown')
                        gt_root = os.path.join(SARENA_ROOT, 'Icon/edit')

                        process_editing_batch(gt_root, task_dir, task_name, edit_calculator)
                        has_processed_edit = True
                        print(f"  [Accumulated] Editing metrics for {task_name}")

                    elif "Generation" in l2_prefix:
                        # Generation Tasks: T2SVG / I2SVG
                        bench_name = category.split("-")[1]
                        task_type = 'text2svg' if 'T2SVG' in l2_prefix else 'img2svg'

                        if bench_name == 'Icon':
                            gt_root = os.path.join(SARENA_ROOT, 'Icon/generation')
                        elif bench_name == 'Illustration':
                            gt_root = os.path.join(SARENA_ROOT, 'illustration')
                        elif bench_name == 'Chemistry':
                            gt_root = os.path.join(SARENA_ROOT, 'chemistry')

                        metrics_res = evaluate_generation(
                            bench_name, task_type, gt_root, task_dir, tokenizer_path
                        )

                        clean_task_name = "T2SVG" if "T2SVG" in l2_prefix else "I2SVG"
                        final_json_results[category][clean_task_name] = metrics_res

                        print(f"  Scores: {metrics_res}")

                except Exception as e:
                    print(f"  Error calculating metrics for {task_key}: {e}")
                    final_json_results[category][l2_prefix]["ERROR"] = str(e)

        # 3. Finalize Editing Metrics
        if has_processed_edit and edit_calculator:
            print("\nCalculating Overall Editing Metrics...")
            overall_edit_score = edit_calculator.summarize_metrics()
            final_json_results["SArena-Icon"]["Editing"] = overall_edit_score
            print(f"Editing Overall Scores: {overall_edit_score}")

    finally:
        pass

    # --- Convert to Regular Dict and Save JSON ---
    def recursive_convert(d):
        if isinstance(d, defaultdict):
            d = {k: recursive_convert(v) for k, v in d.items()}
        elif isinstance(d, dict):
            d = {k: recursive_convert(v) for k, v in d.items()}
        else:
            d = convert_np_to_native(d)
        return d

    final_output_dict = recursive_convert(final_json_results)

    json_output_path = eval_file.replace('.xlsx', '_results.json').replace('.pkl', '_results.json')
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output_dict, f, indent=4, ensure_ascii=False)

    print(f"\n{'=' * 50}\nSArena Evaluation Complete\n{'=' * 50}")
    print(f"Results saved to: {json_output_path}")
    print(json.dumps(final_output_dict, indent=2))

    return final_output_dict
