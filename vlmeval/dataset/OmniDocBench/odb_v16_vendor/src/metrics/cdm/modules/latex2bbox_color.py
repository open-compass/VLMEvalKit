import os
import re
import time
import json
import shutil
import logging
import subprocess
import shlex
import signal
import numpy as np

from PIL import Image, ImageDraw
from .latex_processor import (
    normalize_latex,
    token_add_color_RGB,
    clean_latex
)
from .texlive_env import build_tex_env, resolve_cjk_font_family, resolve_tex_binary
from .tokenize_latex.tokenize_latex import tokenize_latex


tabular_template = r"""
\documentclass[12pt]{article}
\usepackage[landscape]{geometry}
\usepackage{geometry}
\geometry{a<PaperSize>paper,scale=0.98}
\pagestyle{empty}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amssymb}
\usepackage{upgreek}
\usepackage{amsmath}
\usepackage{xcolor}
\begin{document}
\makeatletter
\renewcommand*{\@textcolor}[3]{%%
  \protect\leavevmode
  \begingroup
    \color#1{#2}#3%%
  \endgroup
}
\makeatother
\begin{displaymath}
%s
\end{displaymath}
\end{document}
"""

formular_template = r"""
\documentclass[12pt]{article}
\usepackage[landscape]{geometry}
\usepackage{geometry}
\geometry{a<PaperSize>paper,scale=0.98}
\pagestyle{empty}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{upgreek}
\usepackage{amssymb}
\usepackage{xcolor}
\begin{document}
\makeatletter
\renewcommand*{\@textcolor}[3]{%%
  \protect\leavevmode
  \begingroup
    \color#1{#2}#3%%
  \endgroup
}
\makeatother
\begin{displaymath}
%s
\end{displaymath}
\end{document}
"""

formular_template_zh = r"""
\documentclass[12pt]{article}
\usepackage[landscape]{geometry}
\usepackage{geometry}
\geometry{a<PaperSize>paper,scale=0.98}
\pagestyle{empty}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{upgreek}
\usepackage{CJK}
\usepackage{amssymb}
\usepackage{xcolor}
\begin{document}
\makeatletter
\renewcommand*{\@textcolor}[3]{%%
  \protect\leavevmode
  \begingroup
    \color#1{#2}#3%%
  \endgroup
}
\makeatother
\begin{CJK}{UTF8}{<CJKFont>}
\begin{displaymath}
%s
\end{displaymath}
\end{CJK}
\end{document}
"""


def run_cmd(cmd, timeout_sec=30):
    proc = None
    try:
        preexec_fn = os.setsid if hasattr(os, "setsid") else None
        proc = subprocess.Popen(cmd, shell=True, preexec_fn=preexec_fn, env=build_tex_env())
        proc.communicate(timeout=timeout_sec)
        return int(proc.returncode or 0)
    except subprocess.TimeoutExpired:
        logging.info(f"ERROR, cmd timeout after {timeout_sec}s: {cmd}")
        if proc is not None:
            try:
                if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except Exception:
                pass
        return -1
    except Exception as e:
        logging.info(f"ERROR, cmd failed: {cmd}; {e}.")
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
        return -1


def _tokenize_latex_with_timeout(latex_code: str, middle_file: str) -> tuple[bool, str]:
    """
    Tokenize LaTeX with a hard wall-clock timeout.

    Some pathological formulas can trigger extremely slow parser paths.
    On timeout, return (False, original_latex) so caller can fall back.
    """
    try:
        timeout_sec = float(os.getenv("CDM_TOKENIZE_TIMEOUT_SEC", "30"))
    except Exception:
        timeout_sec = 30.0
    if timeout_sec <= 0:
        return tokenize_latex(latex_code, middle_file=middle_file)
    if not hasattr(signal, "setitimer"):
        return tokenize_latex(latex_code, middle_file=middle_file)

    timed_out = False

    def _handler(signum, frame):  # pragma: no cover - signal callback
        raise TimeoutError("tokenize_latex timeout")

    try:
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout_sec)
    except Exception:
        # Fallback for non-main-thread or unsupported signal envs.
        return tokenize_latex(latex_code, middle_file=middle_file)

    try:
        signal.signal(signal.SIGALRM, _handler)
        return tokenize_latex(latex_code, middle_file=middle_file)
    except TimeoutError:
        timed_out = True
        return False, latex_code
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            # Restore previous timer if there was one.
            if previous_timer[0] > 0 or previous_timer[1] > 0:
                signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])
        except Exception:
            pass
        if timed_out:
            logging.info("ERROR, tokenize latex timeout: %s", middle_file)


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except Exception:
        pass


def _is_image_readable(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


def convert_pdf2img(pdf_filename, png_filename, timeout_sec=30, all_pages=False):
    """
    Convert `pdf_filename` to `png_filename` via ImageMagick.

    - If `all_pages` is False: render only the first page (fast-path).
    - If `all_pages` is True: render and vertically append all pages into one PNG.

    Writes atomically to avoid leaving truncated PNGs under multiprocessing.
    """
    root, ext = os.path.splitext(png_filename)
    tmp_png = f"{root}.tmp.{os.getpid()}.{time.time_ns()}{ext}"
    if all_pages:
        cmd = "magick -density 200 -quality 100 %s -append %s" % (
            shlex.quote(pdf_filename),
            shlex.quote(tmp_png),
        )
    else:
        pdf_spec = f"{pdf_filename}[0]"
        cmd = "magick -density 200 -quality 100 %s %s" % (shlex.quote(pdf_spec), shlex.quote(tmp_png))
    ret = run_cmd(cmd, timeout_sec=timeout_sec)
    if ret != 0:
        _safe_remove(tmp_png)
        return False
    if not os.path.exists(tmp_png) or os.path.getsize(tmp_png) <= 0:
        _safe_remove(tmp_png)
        return False
    if not _is_image_readable(tmp_png):
        _safe_remove(tmp_png)
        return False
    os.replace(tmp_png, png_filename)
    return True


def _parse_pdflatex_log(log_path: str):
    """
    Parse pdflatex log to extract:
    - page_count: int | None
    - has_overfull: bool  (Overfull \\hbox / \\vbox warnings)
    """
    page_count = None
    has_overfull = False
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if page_count is None and "Output written on" in line:
                    m = re.search(r"Output written on .*\((\d+) page", line)
                    if m:
                        try:
                            page_count = int(m.group(1))
                        except Exception:
                            page_count = None
                if not has_overfull and ("Overfull \\hbox" in line or "Overfull \\vbox" in line):
                    has_overfull = True
                if page_count is not None and has_overfull:
                    break
    except Exception:
        return None, False
    return page_count, has_overfull


def crop_image(image_path, pad=8):
    """
    Crop to non-white region and save atomically.
    Returns True on success, False if the image cannot be read/processed.
    """
    try:
        with Image.open(image_path) as img0:
            img_rgb = img0.convert("RGB")
        img_l = img_rgb.convert("L")
        img_data = np.asarray(img_l, dtype=np.uint8)
        nnz_inds = np.where(img_data != 255)
        if len(nnz_inds[0]) == 0:
            x_min, y_min, x_max, y_max = 0, 0, 10, 10
        else:
            y_min = int(np.min(nnz_inds[0]))
            y_max = int(np.max(nnz_inds[0]))
            x_min = int(np.min(nnz_inds[1]))
            x_max = int(np.max(nnz_inds[1]))

        W, H = img_rgb.size
        x0 = max(0, x_min - pad)
        y0 = max(0, y_min - pad)
        x1 = min(W, x_max + pad)
        y1 = min(H, y_max + pad)
        cropped = img_rgb.crop((x0, y0, x1, y1))

        root, ext = os.path.splitext(image_path)
        tmp_path = f"{root}.tmp.{os.getpid()}.{time.time_ns()}{ext}"
        cropped.save(tmp_path)
        os.replace(tmp_path, image_path)
        return True
    except Exception as e:
        logging.info(f"ERROR, crop_image failed: {image_path}; {e}.")
        _safe_remove(image_path)
        return False

def _is_default_color_list_prefix(color_list, gap=15):
    """
    Check whether `color_list` is the prefix of `gen_color_list(num=5800, gap=15)`.
    If true, we can map pixel colors -> token index in O(1) arithmetic instead of per-color full-image scans.
    """
    if not color_list:
        return True
    single_num = 255 // gap + 1  # 18 when gap=15
    stride_r = single_num * single_num  # 324
    stride_g = single_num  # 18

    def expected(idx0: int):
        r = idx0 // stride_r
        gb = idx0 % stride_r
        g = gb // stride_g
        b = gb % stride_g
        return (r * gap, g * gap, b * gap)

    n = len(color_list)
    for idx0 in (1, 2, 3, n):
        if 1 <= idx0 <= n and color_list[idx0 - 1] != expected(idx0):
            return False
    return True


def extrac_bbox_from_color_image_imread(image_path):
    try:
        with Image.open(image_path) as img:
            return np.asarray(img.convert("RGB"))
    except Exception:
        return None


def extrac_bbox_from_color_image_scan(img, color_list, gap=15):
    """
    Extract bbox for each token color.

    Optimized implementation:
    - Fast-path for the default CDM palette (prefix of `gen_color_list(num=5800, gap=15)`):
      one-pass scan of the image and vectorized bbox aggregation.
    - Fallback path for arbitrary color_list: generic mapping by packed color codes.
    """
    num_tokens = len(color_list)
    if num_tokens == 0:
        return []
    if img is None:
        return [[] for _ in range(num_tokens)]

    H, W = img.shape[:2]
    try:
        color_tol = int(os.getenv("CDM_COLOR_TOL", "1"))
    except Exception:
        color_tol = 1
    color_tol = max(0, min(10, color_tol))

    if _is_default_color_list_prefix(color_list, gap=gap):
        # PIL image is RGB; palette is (R,G,B)
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        if color_tol > 0:
            r_i = r.astype(np.int16)
            g_i = g.astype(np.int16)
            b_i = b.astype(np.int16)

            rq = ((r_i + gap // 2) // gap) * gap
            gq = ((g_i + gap // 2) // gap) * gap
            bq = ((b_i + gap // 2) // gap) * gap

            rq = np.clip(rq, 0, 255)
            gq = np.clip(gq, 0, 255)
            bq = np.clip(bq, 0, 255)

            rmask = np.abs(r_i - rq) <= color_tol
            gmask = np.abs(g_i - gq) <= color_tol
            bmask = np.abs(b_i - bq) <= color_tol

            white = (r_i >= 255 - color_tol) & (g_i >= 255 - color_tol) & (b_i >= 255 - color_tol)
            mask = (~white) & rmask & gmask & bmask
            if not mask.any():
                return [[] for _ in range(num_tokens)]

            # Quantize to [0..17] then compute palette idx0 in [0..5831]
            rq = (rq // gap).astype(np.uint16)
            gq = (gq // gap).astype(np.uint16)
            bq = (bq // gap).astype(np.uint16)
        else:
            mask = (b != 255) | (g != 255) | (r != 255)
            if not mask.any():
                return [[] for _ in range(num_tokens)]

            mask &= (b % gap == 0) & (g % gap == 0) & (r % gap == 0)

            # Quantize to [0..17] then compute palette idx0 in [0..5831]
            bq = (b // gap).astype(np.uint16)
            gq = (g // gap).astype(np.uint16)
            rq = (r // gap).astype(np.uint16)
        idx0 = rq * 324 + gq * 18 + bq

        mask &= (idx0 >= 1) & (idx0 <= num_tokens)
        if not mask.any():
            return [[] for _ in range(num_tokens)]

        ys, xs = np.nonzero(mask)
        tok = (idx0[ys, xs] - 1).astype(np.int32)  # 0-based token index

        min_x = np.full(num_tokens, W, dtype=np.int32)
        min_y = np.full(num_tokens, H, dtype=np.int32)
        max_x = np.full(num_tokens, -1, dtype=np.int32)
        max_y = np.full(num_tokens, -1, dtype=np.int32)

        np.minimum.at(min_x, tok, xs)
        np.minimum.at(min_y, tok, ys)
        np.maximum.at(max_x, tok, xs)
        np.maximum.at(max_y, tok, ys)

        bbox_list = []
        for i in range(num_tokens):
            if max_x[i] >= 0:
                bbox_list.append([int(min_x[i] - 1), int(min_y[i] - 1), int(max_x[i] + 1), int(max_y[i] + 1)])
            else:
                bbox_list.append([])
        return bbox_list

    # Fallback: handle arbitrary color_list (slower but correct).
    target_codes = np.array(
        [(r + (g << 8) + (b << 16)) for (r, g, b) in color_list],
        dtype=np.uint32,
    )
    order = np.argsort(target_codes)
    target_sorted = target_codes[order]

    img_codes = (
        img[:, :, 0].astype(np.uint32)
        + (img[:, :, 1].astype(np.uint32) << 8)
        + (img[:, :, 2].astype(np.uint32) << 16)
    )
    flat = img_codes.ravel()
    pos = np.searchsorted(target_sorted, flat)
    valid = pos < target_sorted.shape[0]
    match = np.zeros_like(flat, dtype=bool)
    match[valid] = target_sorted[pos[valid]] == flat[valid]
    if not match.any():
        return [[] for _ in range(num_tokens)]

    flat_idx = np.flatnonzero(match)
    ys = (flat_idx // W).astype(np.int32)
    xs = (flat_idx % W).astype(np.int32)
    tok = order[pos[flat_idx]].astype(np.int32)

    min_x = np.full(num_tokens, W, dtype=np.int32)
    min_y = np.full(num_tokens, H, dtype=np.int32)
    max_x = np.full(num_tokens, -1, dtype=np.int32)
    max_y = np.full(num_tokens, -1, dtype=np.int32)
    np.minimum.at(min_x, tok, xs)
    np.minimum.at(min_y, tok, ys)
    np.maximum.at(max_x, tok, xs)
    np.maximum.at(max_y, tok, ys)

    bbox_list = []
    for i in range(num_tokens):
        if max_x[i] >= 0:
            bbox_list.append([int(min_x[i] - 1), int(min_y[i] - 1), int(max_x[i] + 1), int(max_y[i] + 1)])
        else:
            bbox_list.append([])
    return bbox_list


def extrac_bbox_from_color_image_binarize_save(image_path, img):
    """
    Convert image to pure black/white (keep pure white pixels as white, everything else to black)
    and save back to `image_path`.
    """
    if img is None:
        return
    white = (img[:, :, 0] == 255) & (img[:, :, 1] == 255) & (img[:, :, 2] == 255)
    bw = np.where(white, 255, 0).astype(np.uint8)
    bw3 = np.stack([bw, bw, bw], axis=2)
    Image.fromarray(bw3).save(image_path)


def extrac_bbox_from_color_image(image_path, color_list):
    _t0 = time.perf_counter()
    img = extrac_bbox_from_color_image_imread(image_path)
    _t1 = time.perf_counter()
    bbox_list = extrac_bbox_from_color_image_scan(img, color_list)
    _t2 = time.perf_counter()
    extrac_bbox_from_color_image_binarize_save(image_path, img)
    _t3 = time.perf_counter()
    globals()["_EXTRAC_BBOX_LAST_TIMING"] = {
        "imread": _t1 - _t0,
        "scan": _t2 - _t1,
        "binarize_save": _t3 - _t2,
        "total": _t3 - _t0,
    }
    return bbox_list

def contains_chinese(text):
    return re.search(r'[\u4e00-\u9fff]', text) is not None

def wrap_chinese_in_text(latex_text):
    # Normalize \text { ... } -> \text{...} to avoid double-wrapping Chinese runs.
    latex_text = re.sub(r'\\text\s*\{', r'\\text{', latex_text)
    chinese_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf]'
    chinese_sequence_pattern = chinese_pattern + '+'
    def replace_chinese(match):
        chinese_text = match.group(0)
        start_pos = match.start()
        end_pos = match.end()
        before_text = latex_text[:start_pos]
        after_text = latex_text[end_pos:]
        if re.search(r'\\text\s*\{$', before_text) and re.match(r'\s*}', after_text):
            return chinese_text
        else:
            return f'\\text{{{chinese_text}}}'
    result = re.sub(chinese_sequence_pattern, replace_chinese, latex_text)
    return result


def rewrite_cjk_mathrm_to_text(latex_text):
    """For CJK content, keep text in \\text{...} instead of \\mathrm{...}."""
    if "\\mathrm" not in latex_text:
        return latex_text
    # Tokenizer may output forms like: \mathrm { 传 动 侧 ... }
    pattern = re.compile(r'\\mathrm\s*\{([^{}]*[\u4e00-\u9fff\u3400-\u4dbf][^{}]*)\}')
    return re.sub(pattern, r'\\text{\1}', latex_text)


def rewrite_cjk_text_to_char_level(latex_text):
    """Expand CJK \\text{...} groups to \\text { ... } so each glyph is tokenized separately."""
    if "\\text" not in latex_text:
        return latex_text
    pattern = re.compile(r'\\text\s*\{([^{}]*[\u4e00-\u9fff\u3400-\u4dbf][^{}]*)\}')

    def _repl(match):
        content = match.group(1).strip()
        if not content:
            return match.group(0)
        return r"\text { " + content + " }"

    return re.sub(pattern, _repl, latex_text)


def collapse_cjk_text_groups(tokens):
    """Collapse tokenized \\text{ ... } groups containing CJK into one token."""
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith(r"\text{"):
            out.append(tok)
            i += 1
            continue

        # Find the matching closing "}" in the current tokenized stream.
        j = i + 1
        parts = []
        lead = tok[len(r"\text{") :]
        if lead:
            parts.append(lead)
        while j < len(tokens) and tokens[j] != "}":
            parts.append(tokens[j])
            j += 1

        if j >= len(tokens):
            # Broken group; keep original token.
            out.append(tok)
            i += 1
            continue

        content = "".join(parts)
        if re.search(r"[\u4e00-\u9fff\u3400-\u4dbf]", content):
            out.append(r"\text{" + content + "}")
        else:
            out.extend(tokens[i : j + 1])
        i = j + 1
    return out
    
def latex2bbox_color(input_arg):
    latex, basename, output_path, temp_dir, total_color_list = input_arg
    if "tabular" in latex:
        template = tabular_template
    else:
        if contains_chinese(latex):
            template = formular_template_zh.replace("<CJKFont>", resolve_cjk_font_family())
            latex = latex.replace("，", ", ").replace("：", ": ").replace("；", "; ")
            latex = wrap_chinese_in_text(latex)
        else:
            template = formular_template
    output_bbox_path = os.path.join(output_path, "bbox", basename + ".jsonl")
    output_vis_path = os.path.join(output_path, "vis", basename + ".png")
    output_base_path = os.path.join(output_path, "vis", basename + "_base.png")
    
    if os.path.exists(output_bbox_path) and os.path.exists(output_vis_path) and os.path.exists(output_base_path):
        if _is_image_readable(output_base_path):
            return
        _safe_remove(output_bbox_path)
        _safe_remove(output_vis_path)
        _safe_remove(output_base_path)
    
    try:
        latex = latex.replace("\n", " ")
        latex = latex.replace("\\%", "<PERCENTAGETOKEN>")
        ret, new_latex = _tokenize_latex_with_timeout(
            latex,
            middle_file=os.path.join(temp_dir, basename + ".txt"),
        )
        if not(ret and new_latex):
            log = f"ERROR, Tokenize latex failed: {basename}."
            logging.info(log)
            new_latex = latex
        try:
            cjk_char_level = os.getenv("CDM_CJK_CHAR_LEVEL", "1").strip().lower() not in ("0", "false", "off", "no")
        except Exception:
            cjk_char_level = True
        # Keep Chinese text in \text{...}; \mathrm{CJK} may lose glyphs in some TeX setups.
        if contains_chinese(new_latex):
            new_latex = rewrite_cjk_mathrm_to_text(new_latex)
            if cjk_char_level:
                # Strict mode: evaluate Chinese at per-character granularity.
                new_latex = rewrite_cjk_text_to_char_level(new_latex)
        new_latex = new_latex.replace("< P E R C E N T A G E T O K E N >", "\\%")
        latex = normalize_latex(new_latex)
        token_list = []
        l_split = latex.strip().split(' ')
        if contains_chinese(new_latex) and not cjk_char_level:
            l_split = collapse_cjk_text_groups(l_split)
        color_list = total_color_list[0:len(l_split)]
        idx = 0
        while idx < len(l_split):
            l_split, idx, token_list = token_add_color_RGB(l_split, idx, token_list)

        rgb_latex = " ".join(l_split)
        for idx, color in enumerate(color_list):
            R, G, B = color
            rgb_latex = rgb_latex.replace(f"<color_{idx}>", f"{R},{G},{B}")

        # `paper_size` is injected into LaTeX as `a{paper_size}paper` (ISO A-series):
        # 5 -> A5 (small), 4 -> A4, 3 -> A3 (larger). Smaller numbers mean larger papers (e.g., 1 -> A1).
        if len(token_list) > 1300:
            paper_size = 3
        elif len(token_list) > 600:
            paper_size = 4
        else:
            paper_size = 5
        # Defer final LaTeX assembly until we decide the paper size (may retry with larger papers).
        paper_size_guess = paper_size
        rgb_latex_final = rgb_latex
        template_final = template
        
    except Exception as e:
        log = f"ERROR, Preprocess latex failed: {basename}; {e}."
        logging.info(log)
        return
    
    pre_name = output_path.replace('/', '_').replace('.','_') + '_' + basename
    tex_filename = os.path.join(temp_dir, pre_name+'.tex')
    log_filename = os.path.join(temp_dir, pre_name+'.log')
    aux_filename = os.path.join(temp_dir, pre_name+'.aux')
    
    pdf_filename = tex_filename[:-4] + ".pdf"
    page_count = None
    has_overfull = False

    # Retry with larger paper sizes if pdflatex spills to multiple pages or reports overfull boxes.
    # Default cap is A1 (paper_size=1); override via env `CDM_LATEX_MIN_PAPER_SIZE` (0..5).
    try:
        min_paper_size = int(os.getenv("CDM_LATEX_MIN_PAPER_SIZE", "1"))
    except Exception:
        min_paper_size = 1
    min_paper_size = max(0, min(5, min_paper_size))
    if min_paper_size > paper_size_guess:
        paper_candidates = [paper_size_guess]
    else:
        paper_candidates = list(range(paper_size_guess, min_paper_size - 1, -1))

    for ps in paper_candidates:
        final_latex = template_final.replace("<PaperSize>", str(ps)) % rgb_latex_final
        with open(tex_filename, "w") as w:
            print(final_latex, file=w)
        pdflatex_bin = shlex.quote(resolve_tex_binary("pdflatex"))
        run_cmd(
            f"{pdflatex_bin} -interaction=nonstopmode -output-directory={shlex.quote(temp_dir)} {shlex.quote(tex_filename)} >/dev/null"
        )
        page_count, has_overfull = _parse_pdflatex_log(log_filename)
        if not os.path.exists(pdf_filename):
            break
        need_retry = has_overfull or (page_count is not None and page_count > 1)
        if not need_retry:
            break
        if ps == paper_candidates[-1]:
            break
        _safe_remove(pdf_filename)
        _safe_remove(log_filename)
        _safe_remove(aux_filename)

    if not os.path.exists(pdf_filename):
        log = f"ERROR, Compile pdf failed: {pdf_filename}"
        logging.info(log)
        _safe_remove(tex_filename)
        _safe_remove(log_filename)
        _safe_remove(aux_filename)
        return

    render_all_pages = (page_count is None) or (page_count > 1)
    try:
        ok = convert_pdf2img(pdf_filename, output_base_path, all_pages=render_all_pages)
    finally:
        _safe_remove(pdf_filename)
        _safe_remove(tex_filename)
        _safe_remove(log_filename)
        _safe_remove(aux_filename)

    if not ok:
        logging.info(f"ERROR, pdf2img failed: {pdf_filename} -> {output_base_path}")
        _safe_remove(output_base_path)
        return

    if not crop_image(output_base_path):
        _safe_remove(output_base_path)
        return

    bbox_list = extrac_bbox_from_color_image(output_base_path, color_list)

    root, ext = os.path.splitext(output_bbox_path)
    tmp_bbox = f"{root}.tmp.{os.getpid()}.{time.time_ns()}{ext}"
    with open(tmp_bbox, "w", encoding="utf-8") as f:
        for token, box in zip(token_list, bbox_list):
            item = {"bbox": box, "token": token}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(tmp_bbox, output_bbox_path)

    try:
        with Image.open(output_base_path) as vis:
            vis = vis.convert("RGB")
            draw = ImageDraw.Draw(vis)
            for token, box in zip(token_list, bbox_list):
                if not box:
                    continue
                x_min, y_min, x_max, y_max = box
                draw.rectangle([x_min, y_min, x_max, y_max], fill=None, outline=(0, 250, 0), width=1)
                try:
                    draw.text((x_min, y_min), token, (250, 0, 0))
                except Exception:
                    pass

            root, ext = os.path.splitext(output_vis_path)
            tmp_vis = f"{root}.tmp.{os.getpid()}.{time.time_ns()}{ext}"
            vis.save(tmp_vis)
            os.replace(tmp_vis, output_vis_path)
    except Exception as e:
        logging.info(f"ERROR, save vis failed: {output_vis_path}; {e}.")
