import json
import logging
import multiprocessing
import os
import queue
import re
import traceback
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

from pylatexenc.latex2text import LatexNodes2Text


formula_pattern = re.compile(
    r"\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$|\$.*?\$",
    re.DOTALL,
)

_ENV_RE = re.compile(r"\\(begin|end)\{([^{}]+)\}")
_LEFT_RE = re.compile(r"\\left\b")
_RIGHT_RE = re.compile(r"\\right\b")
_INLINE_OPEN_RE = re.compile(r"\\\(")
_INLINE_CLOSE_RE = re.compile(r"\\\)")
_DISPLAY_OPEN_RE = re.compile(r"\\\[")
_DISPLAY_CLOSE_RE = re.compile(r"\\\]")
_UNESCAPED_DOLLAR_RE = re.compile(r"(?<!\\)\$")
_PYLATEXENC_LOGGER_NAMES = [
    'pylatexenc',
    'pylatexenc.latexwalker',
    'pylatexenc.latexwalker._walker',
    'pylatexenc.macrospec',
    'pylatexenc.macrospec._environmentbodyparser',
]


_LATEX_SIGNAL_RE = re.compile(r"\\[A-Za-z]+|[_^{}$]|\\[\[\]()]]|&|%|#")
_NATURAL_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+")
_LATEX_TIMEOUT_CONTEXT = ContextVar('latex_timeout_context', default=None)


def _get_latex_timeout_context():
    context = _LATEX_TIMEOUT_CONTEXT.get()
    if not context:
        return {}
    return dict(context)


def _format_latex_timeout_context(context=None):
    context = _get_latex_timeout_context() if context is None else dict(context or {})
    if not context:
        return ''
    parts = []
    img_name = context.get('img_name')
    pred_path = context.get('pred_path')
    if img_name:
        parts.append(f'img_name={json.dumps(img_name, ensure_ascii=False)}')
    if pred_path:
        parts.append(f'pred_path={json.dumps(pred_path, ensure_ascii=False)}')
    return ' '.join(parts)


@contextmanager
def latex_timeout_context(img_name=None, pred_path=None):
    current = _get_latex_timeout_context()
    if img_name is not None:
        current['img_name'] = img_name
    if pred_path is not None:
        current['pred_path'] = pred_path
    token = _LATEX_TIMEOUT_CONTEXT.set(current)
    try:
        yield
    finally:
        _LATEX_TIMEOUT_CONTEXT.reset(token)


def _read_float_env(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _resolve_timeout_input_dir():
    return os.getenv('OMNIDOCBENCH_TIMEOUT_INPUT_DIR', os.path.join(os.getcwd(), 'logs', 'timeout_inputs'))


def _dump_timeout_input(prefix, payload):
    try:
        out_dir = _resolve_timeout_input_dir()
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{prefix}_{uuid.uuid4().hex}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return out_path
    except Exception as exc:
        logging.warning(f'ERROR: failed to dump timeout input: {exc}')
        return ''


def _log_latex_timeout_input(reason, latex, timeout=None, error=None):
    context = _get_latex_timeout_context()
    payload = {
        'stage': 'latex_to_text',
        'reason': reason,
        'timeout_sec': timeout,
        'error': error,
        'input': str(latex or ''),
        'img_name': context.get('img_name'),
        'pred_path': context.get('pred_path'),
    }
    dump_path = _dump_timeout_input('latex_to_text', payload)
    preview_limit = max(200, _read_int_env('OMNIDOCBENCH_TIMEOUT_INPUT_PRINT_MAX_CHARS', 1500))
    input_text = str(latex or '')
    if len(input_text) <= preview_limit:
        preview = input_text
    else:
        preview = input_text[:preview_limit] + ' ... [truncated in log, full input saved]'
    context_text = _format_latex_timeout_context(context)
    context_segment = f' {context_text}' if context_text else ''
    print(
        f'[latex-to-text-timeout-input] reason={reason} timeout={timeout}{context_segment} path={dump_path} input={json.dumps(preview, ensure_ascii=False)}',
        flush=True,
    )


def looks_like_plaintext_formula_noise(line):
    stripped = str(line or '').strip()
    if len(stripped) < 32:
        return False
    signal_count = len(_LATEX_SIGNAL_RE.findall(stripped))
    word_count = len(_NATURAL_WORD_RE.findall(stripped))
    if signal_count > 2:
        return False
    if '{' in stripped or '}' in stripped or '$' in stripped or '^' in stripped or '_' in stripped:
        return False
    if any(token in stripped for token in ['\\frac', '\\sqrt', '\\left', '\\right', '\\begin', '\\end', '\\sum', '\\int', '\\alpha', '\\beta']):
        return False
    return word_count >= 6 and stripped.count(' ') >= 5


def looks_like_weak_latex_input(line, latex_type='formula'):
    stripped = str(line or '').strip()
    if not stripped:
        return True

    if stripped in {'\\', '\\\\'}:
        return True

    command_hits = len(_LATEX_COMMAND_RE.findall(stripped))
    word_hits = len(_NATURAL_WORD_RE.findall(stripped))
    backslash_hits = stripped.count('\\')
    math_signal_hits = sum(ch in '{}$^_=&%#' for ch in stripped)

    if stripped.endswith('\\') and not stripped.endswith('\\\\') and command_hits == 0:
        return True

    if backslash_hits > 0 and command_hits == 0 and math_signal_hits == 0:
        return True

    if latex_type in {'formula', 'inline', 'display'} and word_hits >= 2 and command_hits == 0 and math_signal_hits <= 1:
        return True

    if latex_type in {'formula', 'inline', 'display'} and word_hits >= 4 and command_hits <= 1 and math_signal_hits <= 2:
        return True

    return False


def _read_int_env(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)



def _resolve_timeout(timeout=None, env_name='OMNIDOCBENCH_LATEX_TO_TEXT_TIMEOUT_SEC', default=30.0):
    if timeout is not None:
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            timeout = default
    else:
        try:
            timeout = float(os.getenv(env_name, str(default)))
        except Exception:
            timeout = default
    if timeout <= 0:
        return None
    return timeout



def braces_balanced(line):
    depth = 0
    idx = 0
    while idx < len(line):
        ch = line[idx]
        if ch == "\\" and idx + 1 < len(line) and line[idx + 1] in "{}[]()":
            idx += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
        idx += 1
    return depth == 0



def env_balanced(line):
    stack = []
    for match in _ENV_RE.finditer(line):
        kind, name = match.group(1), match.group(2)
        if kind == 'begin':
            stack.append(name)
            continue
        if not stack or stack[-1] != name:
            return False
        stack.pop()
    return not stack



def _math_delims_balanced(line):
    if len(_INLINE_OPEN_RE.findall(line)) != len(_INLINE_CLOSE_RE.findall(line)):
        return False
    if len(_DISPLAY_OPEN_RE.findall(line)) != len(_DISPLAY_CLOSE_RE.findall(line)):
        return False
    dollar_count = len(_UNESCAPED_DOLLAR_RE.findall(line))
    if dollar_count % 2 != 0:
        return False
    return True



def _left_right_balanced(line):
    return len(_LEFT_RE.findall(line)) == len(_RIGHT_RE.findall(line))



def likely_bad_latex(line, latex_type='formula'):
    stripped = str(line or '').strip()
    if not stripped:
        return False, ''

    max_len = _read_int_env('OMNIDOCBENCH_MAX_LATEX_INPUT_LEN', 8000)
    if max_len > 0 and len(stripped) > max_len:
        return True, f'too_long:{len(stripped)}'

    if not braces_balanced(stripped):
        return True, 'unbalanced_braces'

    if not env_balanced(stripped):
        return True, 'unbalanced_env'

    if latex_type in {'formula', 'inline', 'display'}:
        if not _left_right_balanced(stripped):
            return True, 'unbalanced_left_right'
        if not _math_delims_balanced(stripped):
            return True, 'unbalanced_math_delims'

    return False, ''


@contextmanager
def suppress_pylatexenc_warnings():
    logger_states = []
    for name in _PYLATEXENC_LOGGER_NAMES:
        logger = logging.getLogger(name)
        logger_states.append((logger, logger.level))
        logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for logger, level in logger_states:
            logger.setLevel(level)



def clean_string(input_string):
    input_string = str(input_string or '')
    input_string = (
        input_string.replace("\\t", "")
        .replace("\\n", "")
        .replace("\t", "")
        .replace("\n", "")
        .replace("/t", "")
        .replace("/n", "")
        .replace(" ", "")
        .replace("✓", "✔")
        .replace("√", "✔")
        .replace("-", "—")
        .replace("∼", "～")
        .replace("Ø", "∅")
    )
    input_string = re.sub(r"_{4,}", "____", input_string)
    input_string = re.sub(r" {4,}", "    ", input_string)
    return input_string



def remove_math_font(latex):
    filter_list = [
        "\\mathbf",
        "\\mathrm",
        "\\mathnormal",
        "\\mathit",
        "\\mathbb",
        "\\mathcal",
        "\\mathscr",
        "\\mathfrak",
        "\\mathsf",
        "\\mathtt",
        "\\textbf",
        "\\textit",
    ]
    for filter_text in filter_list:
        latex = latex.replace(filter_text, "")
    return latex



def _latex_to_text_worker(latex, result_queue):
    try:
        with suppress_pylatexenc_warnings():
            result = LatexNodes2Text().latex_to_text(latex)
        result_queue.put(('success', result))
    except Exception as exc:
        trace_string = traceback.format_exc()
        result_queue.put(('error', f'{type(exc)}: {exc}, {trace_string}'))



def _latex_to_text_with_timeout(latex, timeout=None):
    if looks_like_plaintext_formula_noise(latex):
        return str(latex or '').strip()
    timeout = _resolve_timeout(timeout)
    if timeout is None:
        try:
            with suppress_pylatexenc_warnings():
                return LatexNodes2Text().latex_to_text(latex)
        except Exception as exc:
            context_suffix = _format_latex_timeout_context()
            context_suffix = f" [{context_suffix}]" if context_suffix else ""
            logging.warning(f"ERROR: latex_to_text failed{context_suffix}: {exc}, >>>{str(latex)[:100]}...<<<")
            _log_latex_timeout_input('failed_no_timeout', latex, timeout=None, error=str(exc))
            return None

    result_queue = multiprocessing.Queue()
    result_queue.cancel_join_thread()
    process = multiprocessing.Process(target=_latex_to_text_worker, args=(latex, result_queue))
    process.daemon = True
    try:
        process.start()
        status, result = result_queue.get(timeout=timeout)
        if status == 'success':
            return result
        if status == 'error':
            context_suffix = _format_latex_timeout_context()
            context_suffix = f" [{context_suffix}]" if context_suffix else ""
            logging.warning(f"ERROR: latex_to_text failed{context_suffix}: {result}, >>>{str(latex)[:100]}...<<<")
            _log_latex_timeout_input('worker_error', latex, timeout=timeout, error=result)
    except queue.Empty:
        context_suffix = _format_latex_timeout_context()
        context_suffix = f" [{context_suffix}]" if context_suffix else ""
        logging.warning(f"ERROR: latex_to_text exceeded {timeout}s{context_suffix}, >>>{str(latex)[:100]}...<<<")
        _log_latex_timeout_input('timeout', latex, timeout=timeout)
    except Exception as exc:
        context_suffix = _format_latex_timeout_context()
        context_suffix = f" [{context_suffix}]" if context_suffix else ""
        logging.warning(f"ERROR: latex_to_text failed{context_suffix}: {exc}, >>>{str(latex)[:100]}...<<<")
        _log_latex_timeout_input('failed', latex, timeout=timeout, error=str(exc))
    finally:
        process.join(timeout=1)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
        try:
            result_queue.close()
        except Exception:
            pass
    return None



def safe_latex_to_text(latex, timeout=None, fallback=None, latex_type='formula'):
    stripped = str(latex or '').strip()
    if not stripped:
        return '' if fallback is None else fallback

    disable_flag = str(os.getenv('OMNIDOCBENCH_DISABLE_LATEX_TO_TEXT', '')).strip().lower()
    if disable_flag in {'1', 'true', 'yes', 'on'}:
        return stripped if fallback is None else fallback

    if '\\' not in stripped and '{' not in stripped and '}' not in stripped and '$' not in stripped:
        return stripped

    if looks_like_plaintext_formula_noise(stripped):
        return stripped if fallback is None else fallback

    if looks_like_weak_latex_input(stripped, latex_type=latex_type):
        return stripped if fallback is None else fallback

    is_bad, _ = likely_bad_latex(stripped, latex_type=latex_type)
    if is_bad:
        return stripped if fallback is None else fallback

    text = _latex_to_text_with_timeout(stripped, timeout=timeout)
    if text is None:
        return stripped if fallback is None else fallback
    return text



def latex2unicode(latex):
    latex = str(latex or '').strip()
    if latex.startswith(("\\(", "\\[")):
        latex = latex[2:]
    if latex.endswith(("\\)", "\\]")):
        latex = latex[:-2]
    latex = latex.strip('$')
    latex = remove_math_font(latex)
    latex = (
        latex.replace("\checkmark", "✔")
        .replace("\pm", "±")
        .replace(r"\alpha", "α")
        .replace(r"\beta", "β")
        .replace(r"\gama", "γ")
        .replace(r"\gamma", "γ")
        .replace(r"\mu", "μ")
        .replace(r"\lambda", "λ")
        .replace(r"\theta", "θ")
        .replace(r"\eta", "η")
        .replace(r"\pi", "π")
        .replace(r"\rho", "ρ")
        .replace(r"\sigma", "σ")
        .replace(r"\omega", "ω")
        .replace(r"\delta", "δ")
        .replace("∼", "～")
        .replace("-", "—")
        .replace("Ø", "∅")
    )

    unicode_text = safe_latex_to_text(latex, fallback=latex, latex_type='formula')
    unicode_text = unicode_text.replace(' ', '')
    unicode_text = unicode_text.replace('\xa0', '')
    return unicode_text



def text_post_process(text):
    text = str(text or '')
    parts = []
    last_end = 0
    for match in formula_pattern.finditer(text):
        parts.append(clean_string(text[last_end:match.start()]))
        parts.append(latex2unicode(text[match.start():match.end()]))
        last_end = match.end()
    parts.append(clean_string(text[last_end:]))
    return ''.join(parts)
