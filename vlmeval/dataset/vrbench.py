import hashlib
import json
import os
import os.path as osp
import re
import shutil
import struct
import subprocess
import threading
import time
import warnings
import zipfile
import zlib
from bisect import bisect_right
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from string import Template

import cv2
import numpy as np
import pandas as pd
import portalocker
from PIL import Image

from vlmeval.smp import LMUDataRoot, dump, get_file_extension, get_intermediate_file_path, load
from vlmeval.utils import track_progress_rich
from .video_base import VideoBaseDataset

FAIL_MSG = 'Failed to obtain answer via API.'

VIDEO_ARCHIVE_FILES = [
    *[f'v001_360p_zips/v001_360p.z{i:02d}' for i in range(1, 40)],
    'v001_360p_zips/v001_360p.zip',
]

MCQ_COT_PROMPT = Template(
    '\n'
    'You are a helpful video understanding assistant that answers multi-choice questions through '
    'step-by-step reasoning based on the video and its summary.\n'
    '\n'
    '# Instructions:\n'
    '1. Break down the reasoning process into clear, specific events.\n'
    '2. Conclude with the best option letter(A/B/C/D) at last.\n'
    '\n'
    '# Output Format:\n'
    '<Step 1> Description of event/observation\n'
    '<Step 2> Description of event/observation\n'
    '...\n'
    '<Answer> [Option letter]\n'
    '\n'
    '# Multiple Choice Question\n'
    '$multiple_choice_question\n'
    '\n'
    '# Video Summary\n'
    '$video_summary\n'
)

MCQ_THINK_PROMPT = Template(
    'You are a helpful video understanding assistant. Answer the following multiple-choice '
    'question based on the video and its summary. Output only the option letter (A/B/C/D).\n'
    '\n'
    '# Multiple Choice Question\n'
    '$multiple_choice_question\n'
    '\n'
    '# Video Summary\n'
    '$video_summary\n'
)

UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT = (
    '\n'
    'You are a reasoning process evaluation model. Given the question, your task is to compare '
    "the model's reasoning process with the correct reasoning process provided, and assess the "
    "accuracy of the model's reasoning.\n"
    'Based on the number of correct steps and the overall correctness, give a score between 0 and '
    '10, where 10 means fully correct.\n'
    '\n'
    'Evaluation Criteria:\n'
    '    1. **Step-by-Step Match**: How closely each reasoning step aligns with the ground truth '
    'process. Highest weight (40%).\n'
    '    2. **Logical Integrity**: Whether the reasoning maintains valid logical progression and '
    'complete argumentation (30%).\n'
    '    3. **Factual Correctness**: Absence of factual errors conflicting with established '
    'truths (20%).\n'
    '    4. **Process Clarity**: Clear articulation and organization of reasoning steps (10%).\n'
    '\n'
    '    Scoring:\n'
    '    - **0-3**: Multiple missing/critical deviations from correct steps (≤30% match), broken '
    'logic, severe factual errors, or incoherent presentation.\n'
    '    - **4-6**: Partial step alignment (40-60% match), basic logical structure with gaps, '
    'minor factual slips, or ambiguous explanations.\n'
    '    - **7-9**: Majority steps correct (70-90% match), sound logic with minor jumps, '
    'near-perfect factual accuracy, and clear presentation.\n'
    '    - **10**: Full step correspondence (100% match), flawless logic, perfect factual '
    'accuracy, and exceptionally clear reasoning flow.\n'
    '\n'
    'Please provide the reasons for your scoring at the end.\n'
    'Output Format:\n'
    '<rate>the score (0-10)</rate>.\n'
    '<reason>Briefly explain the reason for the score.</reason>\n'
)

UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT_SEPARATE = (
    '\n'
    'You are a reasoning process evaluation model.\n'
    "Given the question, compare the model's reasoning process with the correct reference process "
    'and evaluate across four dimensions.\n'
    'Provide separate scores (0-10) per dimension with specific evaluation standards.\n'
    'Provide key and very brief explanation of the scores.\n'
    '\n'
    'Evaluation Dimensions:\n'
    '    1. **Step Matching** (0-10):\n'
    '    - Evaluate alignment of reasoning steps with the reference reasoning process\n'
    '    - Detect omissions of critical steps or redundant additions\n'
    '    - Verify completeness of problem decomposition and sequence validity\n'
    '\n'
    '    2. **Logical Consistency** (0-10):\n'
    '    - Validate causal connections in the reasoning chain\n'
    '    - Identify logical leaps or argument discontinuities\n'
    '    - Assess congruence between assumptions and conclusions\n'
    '\n'
    '    3. **Factual Accuracy** (0-10):\n'
    '    - Verify verifiability of all factual claims\n'
    '    - Detect conflicts with established truths\n'
    '    - Evaluate frequency and impact of factual errors\n'
    '\n'
    '    4. **Process Clarity** (0-10):\n'
    '    - Analyze clarity and organization of step presentation\n'
    '    - Check terminology accuracy and consistency\n'
    '    - Assess effectiveness in explaining complex concepts\n'
    '\n'
    'Scoring Standards:\n'
    '    For each dimension:\n'
    '    9-10: Exemplary performance with no flaws\n'
    '    7-8: Non-critical deviations present\n'
    '    5-6: Quality-impairing defects\n'
    '    3-4: Serious validity-compromising errors\n'
    '    0-2: Fundamental functionality failure\n'
    '\n'
    'Output Format:\n'
    '<step_matching>[0-10]</step_matching>\n'
    '<logical_consistency>[0-10]</logical_consistency>\n'
    '<factual_accuracy>[0-10]</factual_accuracy>\n'
    '<process_clarity>[0-10]</process_clarity>\n'
    '<rationale>\n'
    '    [Per-dimension basis:\n'
    '    - Highlight alignment strengths\n'
    '    - Specify critical deficiencies]\n'
    '</rationale>\n'
)

NON_UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT = (
    '\n'
    'You are a reasoning process evaluation model. Given the question, your task is to evaluate '
    "the model's reasoning process with the correct reasoning process provided, and assess the "
    "accuracy of the model's reasoning.\n"
    'In addition to referring to the provided reasoning process and result, you may also assess '
    "the reasoning's validity based on the video summary. If the reasoning is logical and the "
    'result is reasonable, you can adjust the score accordingly.\n'
    'Based on the correctness and reasonableness of the reasoning process, provide a score '
    'between 0 and 10, where 10 means fully correct and reasonable.\n'
    '\n'
    'Evaluation Criteria:\n'
    '    1. **Relevance and Completeness**: Evaluate whether the reasoning process adequately '
    'addresses the question and covers all essential steps, even if the approach differs from the '
    'provided standard. (40%)\n'
    '    2. **Logical Consistency**: Assess the logical progression, coherence, and structural '
    'integrity of the reasoning process (30%).\n'
    '    3. **Factual Accuracy**: Check for correctness and the absence of significant factual '
    'errors (20%).\n'
    '    4. **Clarity and Persuasiveness**: Consider the clarity, organization, and '
    'persuasiveness of the reasoning, including the explanation of alternative valid approaches '
    '(10%).\n'
    '\n'
    'Scoring:\n'
    '    - **0-3**: The reasoning process shows significant omissions, major logical '
    'inconsistencies, or severe factual errors, resulting in an unclear and unconvincing '
    'explanation.\n'
    '    - **4-6**: The reasoning process partially addresses the question with some logical or '
    'factual issues, and the explanation may be somewhat ambiguous or incomplete.\n'
    '    - **7-9**: The reasoning process is largely relevant and logically consistent, with '
    'minor issues in clarity or factual details, leading to a well-argued explanation.\n'
    '    - **10**: The reasoning process fully addresses the question with impeccable logic, '
    'complete factual accuracy, and is presented in a clear and highly persuasive manner.\n'
    '\n'
    'Output Format:\n'
    '<rate>the score (0-10)</rate>.\n'
    '<reason>Briefly explain the reason for the score.</reason>\n'
)

NON_UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT_SEPARATE = (
    '\n'
    'You are a reasoning process evaluation model.\n'
    "Given the question, your task is to evaluate the model's reasoning process with the provided "
    'correct reasoning process and video summary across four distinct dimensions.\n'
    'Provide separate scores (0-10) for each criterion based on its specific evaluation '
    'standards.\n'
    'Provide key and very brief explanation of the scores.\n'
    '\n'
    'Evaluation Dimensions:\n'
    '    1. **Relevance** (0-10):\n'
    '    - Assess whether the reasoning process closely relates to the provided correct reasoning '
    'process\n'
    '    - Evaluate whether the reasoning process fully addresses the question requirements\n'
    '    - Consider handling of edge cases and alternative approaches\n'
    '\n'
    '    2. **Logical Consistency** (0-10):\n'
    '    - Examine the coherence between reasoning steps\n'
    '    - Verify absence of contradictions or fallacies\n'
    '    - Assess structural integrity and progression validity\n'
    '\n'
    '    3. **Factual Accuracy** (0-10):\n'
    '    - Verify correctness of factual claims in the provided\n'
    '    - Check consistency with provided reference correct reasoning process and video summary\n'
    '    - Evaluate error frequency and severity\n'
    '\n'
    '    4. **Clarity and Persuasiveness** (0-10):\n'
    '    - Assess explanation clarity and organization\n'
    '    - Evaluate effectiveness of supporting evidence\n'
    '    - Consider presentation logic and accessibility\n'
    '\n'
    'Dimension Scoring Standards:\n'
    '    For each dimension:\n'
    '    9-10: Exemplary performance with no notable issues\n'
    '    7-8:  Minor imperfections with negligible impact\n'
    '    5-6:  Moderate issues affecting quality\n'
    '    3-4:  Serious deficiencies impairing validity\n'
    '    0-2:  Fundamental failures in this dimension\n'
    '\n'
    'Output Format:\n'
    '<relevance>[0-10]</relevance>\n'
    '<logical_consistency>[0-10]</logical_consistency>\n'
    '<factual_accuracy>[0-10]</factual_accuracy>\n'
    '<clarity>[0-10]</clarity>\n'
    '<rationale>\n'
    '    [For each dimension with simply explanation:\n'
    '    - Key strengths identified\n'
    '    - Specific weaknesses noted]\n'
    '</rationale>\n'
)

UNIQUE_ANSWER_EVAL_HUMAN_PROMPT_TEMPLATE = (
    '\n'
    '# Question\n'
    '{question}\n'
    "# Model's reasoning process and Answer\n"
    '{response}\n'
    '# Correct reasoning step and Answer\n'
    'Reasoning Step:\n'
    '{procedure}\n'
    'Answer:\n'
    '{answer}\n'
    'Please provide your rating and brief reasons.\n'
)

NON_UNIQUE_ANSWER_EVAL_HUMAN_PROMPT_TEMPLATE = (
    '\n'
    '# Video Summary\n'
    '{video_summary}\n'
    '# Question\n'
    '{question}\n'
    "# Model's reasoning process and Answer\n"
    '{response}\n'
    '# Correct reasoning step and Answer\n'
    'Answer:\n'
    '{answer}\n'
    'Reasoning Step:\n'
    '{procedure}\n'
    'Please provide your rating and brief reasons.\n'
)

UNIQUE_TYPES = {'Event Attribution', 'Multi-element Inference', 'Implicit Inference', 'Logical Linkage'}
NON_UNIQUE_TYPES = {'Hypothetical Reasoning', 'Event Prediction'}


def _loads(value, fallback):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return fallback
    return value


def _ensure_hf_cache_env(dataset_root):
    cache_root = osp.join(dataset_root, '.cache', 'huggingface')
    hub_cache = osp.join(cache_root, 'hub')
    xet_cache = osp.join(cache_root, 'xet')
    os.makedirs(hub_cache, exist_ok=True)
    os.makedirs(xet_cache, exist_ok=True)

    os.environ.setdefault('HF_HOME', cache_root)
    os.environ.setdefault('HF_HUB_CACHE', hub_cache)
    os.environ.setdefault('HF_XET_CACHE', xet_cache)

    try:
        from huggingface_hub import constants
        constants.HF_HOME = os.environ['HF_HOME']
        constants.HF_HUB_CACHE = os.environ['HF_HUB_CACHE']
        constants.HF_XET_CACHE = os.environ['HF_XET_CACHE']
    except Exception:
        pass


def _coalesce_ranges(ranges, expected_size):
    cleaned = []
    for start, end in ranges:
        start = max(0, int(start))
        end = min(expected_size, int(end))
        if start < end:
            cleaned.append((start, end))
    cleaned.sort()
    merged = []
    for start, end in cleaned:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _range_size(ranges):
    return sum(end - start for start, end in ranges)


def _load_completed_ranges(manifest_path, filename, expected_size, part_size):
    if manifest_path.exists():
        with manifest_path.open('r', encoding='utf-8') as f:
            manifest = json.load(f)
        if (
            manifest.get('filename') == filename
            and manifest.get('expected_size') == expected_size
        ):
            return _coalesce_ranges(manifest.get('completed', []), expected_size)
    if 0 < part_size < expected_size:
        return [(0, part_size)]
    return []


def _save_completed_ranges(manifest_path, filename, expected_size, completed):
    tmp_path = manifest_path.with_name(manifest_path.name + '.tmp')
    payload = {
        'filename': filename,
        'expected_size': expected_size,
        'completed': completed,
    }
    with tmp_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f)
    os.replace(tmp_path, manifest_path)


def _missing_ranges(expected_size, completed, chunk_size):
    gaps = []
    cursor = 0
    for start, end in completed:
        if cursor < start:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < expected_size:
        gaps.append((cursor, expected_size))

    chunks = []
    for start, end in gaps:
        cursor = start
        while cursor < end:
            next_cursor = min(end, cursor + chunk_size)
            chunks.append((cursor, next_cursor))
            cursor = next_cursor
    return chunks


def _env_int(name, default, minimum=1):
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return max(minimum, int(value))
    except ValueError:
        warnings.warn(f'Ignoring invalid integer env {name}={value!r}.')
        return default


def _download_http_range(url, headers, part_path, start, end, progress, progress_lock):
    import httpx
    from huggingface_hub.utils import hf_raise_for_status

    offset = start
    last_error = None
    timeout_value = _env_int('VRBENCH_DOWNLOAD_TIMEOUT', 60)
    timeout = httpx.Timeout(
        timeout_value,
        connect=timeout_value,
        read=timeout_value,
        write=timeout_value,
        pool=timeout_value,
    )
    for attempt in range(1, 11):
        range_headers = dict(headers)
        range_headers['Range'] = f'bytes={offset}-{end - 1}'
        try:
            with httpx.stream(
                'GET',
                url,
                headers=range_headers,
                follow_redirects=True,
                timeout=timeout,
            ) as response:
                if response.status_code != 206:
                    hf_raise_for_status(response)
                    raise OSError(
                        f'HTTP server ignored Range request for bytes {offset}-{end - 1}: '
                        f'status={response.status_code}'
                    )
                fd = os.open(part_path, os.O_WRONLY)
                try:
                    for block in response.iter_bytes(chunk_size=1024 * 1024):
                        if not block:
                            continue
                        if offset + len(block) > end:
                            block = block[:end - offset]
                        os.pwrite(fd, block, offset)
                        offset += len(block)
                        if progress is not None:
                            with progress_lock:
                                progress.update(len(block))
                        if offset >= end:
                            return
                finally:
                    os.close(fd)
            if offset >= end:
                return
        except Exception as e:
            last_error = e
            time.sleep(min(8, attempt))
    raise OSError(f'Failed to download byte range {start}-{end - 1} for {part_path}') from last_error


def _download_with_http_ranges(url, headers, dst_path, filename, expected_size):
    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    part_path = dst_path.with_name(dst_path.name + '.part')
    manifest_path = dst_path.with_name(dst_path.name + '.part.json')
    part_size = part_path.stat().st_size if part_path.exists() else 0
    completed = _load_completed_ranges(manifest_path, filename, expected_size, part_size)
    completed = _coalesce_ranges(completed, expected_size)

    with part_path.open('ab'):
        pass
    with part_path.open('r+b') as f:
        f.truncate(expected_size)

    workers = _env_int('VRBENCH_DOWNLOAD_WORKERS', 8)
    chunk_size = _env_int('VRBENCH_DOWNLOAD_CHUNK_BYTES', 16 * 1024 * 1024, minimum=1024 * 1024)
    missing = _missing_ranges(expected_size, completed, chunk_size)
    completed_bytes = _range_size(completed)
    print(
        f'VRBench archive HTTP range state: {filename} '
        f'{completed_bytes}/{expected_size} bytes complete, '
        f'{len(missing)} ranges pending, workers={workers}',
        flush=True,
    )
    if not missing:
        os.replace(part_path, dst_path)
        manifest_path.unlink(missing_ok=True)
        return

    progress_lock = threading.Lock()
    progress = None
    if tqdm is not None:
        progress = tqdm(
            total=expected_size,
            initial=completed_bytes,
            unit='B',
            unit_scale=True,
            desc=filename,
        )
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _download_http_range,
                    url,
                    headers,
                    str(part_path),
                    start,
                    end,
                    progress,
                    progress_lock,
                ): (start, end)
                for start, end in missing
            }
            for future in as_completed(futures):
                start, end = futures[future]
                future.result()
                with progress_lock:
                    completed.append((start, end))
                    completed = _coalesce_ranges(completed, expected_size)
                    _save_completed_ranges(manifest_path, filename, expected_size, completed)
    finally:
        if progress is not None:
            progress.close()

    completed = _coalesce_ranges(completed, expected_size)
    if completed != [(0, expected_size)]:
        raise OSError(f'VRBench archive download incomplete: {part_path}')
    os.replace(part_path, dst_path)
    manifest_path.unlink(missing_ok=True)


class _ConcatBinaryFile:

    def __init__(self, paths):
        self.paths = [Path(path) for path in paths]
        self.files = [open(path, 'rb') for path in self.paths]
        self.sizes = [path.stat().st_size for path in self.paths]
        self.starts = []
        offset = 0
        for size in self.sizes:
            self.starts.append(offset)
            offset += size
        self.total_size = offset
        self.pos = 0
        self.name = '+'.join(str(path) for path in self.paths)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            new_pos = offset
        elif whence == os.SEEK_CUR:
            new_pos = self.pos + offset
        elif whence == os.SEEK_END:
            new_pos = self.total_size + offset
        else:
            raise ValueError(f'Unsupported whence: {whence}')
        if new_pos < 0:
            raise ValueError(f'Negative seek position: {new_pos}')
        self.pos = min(new_pos, self.total_size)
        return self.pos

    def read(self, size=-1):
        if self.pos >= self.total_size:
            return b''
        if size is None or size < 0:
            size = self.total_size - self.pos
        remaining = min(size, self.total_size - self.pos)
        chunks = []
        while remaining > 0:
            file_idx = bisect_right(self.starts, self.pos) - 1
            file_offset = self.pos - self.starts[file_idx]
            to_read = min(remaining, self.sizes[file_idx] - file_offset)
            self.files[file_idx].seek(file_offset)
            chunk = self.files[file_idx].read(to_read)
            if not chunk:
                break
            chunks.append(chunk)
            chunk_len = len(chunk)
            self.pos += chunk_len
            remaining -= chunk_len
        return b''.join(chunks)

    def close(self):
        for file in self.files:
            file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _patch_zipfile_multidisk_for_concat():
    """Allow zipfile to read a split archive presented as one concatenated file.

    Python's zipfile rejects ZIP64 archives whose end records say they span
    multiple disks. For VRBench we pass all split parts through
    _ConcatBinaryFile, so the underlying reader is effectively a single file.
    This patch is scoped to the process and only relaxes the disk-number check.
    """
    if getattr(zipfile, '_vrbench_multidisk_patch', False):
        return

    original_endrec = zipfile._EndRecData

    def patched_endrec64(fpin, offset, endrec):
        offset -= zipfile.sizeEndCentDir64Locator
        if offset < 0:
            return endrec
        fpin.seek(offset)
        data = fpin.read(zipfile.sizeEndCentDir64Locator)
        if len(data) != zipfile.sizeEndCentDir64Locator:
            raise OSError('Unknown I/O error')
        sig, _, reloff, _ = struct.unpack(zipfile.structEndArchive64Locator, data)
        if sig != zipfile.stringEndArchive64Locator:
            return endrec

        offset -= zipfile.sizeEndCentDir64
        if reloff > offset:
            raise zipfile.BadZipFile('Corrupt zip64 end of central directory locator')

        fpin.seek(reloff)
        extrasz = offset - reloff
        data = fpin.read(zipfile.sizeEndCentDir64)
        if len(data) != zipfile.sizeEndCentDir64:
            raise OSError('Unknown I/O error')
        if not data.startswith(zipfile.stringEndArchive64) and reloff != offset:
            fpin.seek(offset)
            extrasz = 0
            data = fpin.read(zipfile.sizeEndCentDir64)
            if len(data) != zipfile.sizeEndCentDir64:
                raise OSError('Unknown I/O error')
        if not data.startswith(zipfile.stringEndArchive64):
            raise zipfile.BadZipFile('Zip64 end of central directory record not found')

        sig, sz, _, _, _, _, dircount, dircount2, dirsize, diroffset = struct.unpack(
            zipfile.structEndArchive64,
            data,
        )
        if diroffset + dirsize != reloff or sz + 12 != zipfile.sizeEndCentDir64 + extrasz:
            raise zipfile.BadZipFile('Corrupt zip64 end of central directory record')

        endrec[zipfile._ECD_SIGNATURE] = sig
        endrec[zipfile._ECD_DISK_NUMBER] = 0
        endrec[zipfile._ECD_DISK_START] = 0
        endrec[zipfile._ECD_ENTRIES_THIS_DISK] = dircount2
        endrec[zipfile._ECD_ENTRIES_TOTAL] = dircount2
        endrec[zipfile._ECD_SIZE] = dirsize
        endrec[zipfile._ECD_OFFSET] = diroffset
        endrec[zipfile._ECD_LOCATION] = offset - extrasz
        return endrec

    def patched_endrec(fpin):
        zipfile._EndRecData64 = patched_endrec64
        endrec = original_endrec(fpin)
        if endrec is not None:
            endrec[zipfile._ECD_DISK_NUMBER] = 0
            endrec[zipfile._ECD_DISK_START] = 0
            endrec[zipfile._ECD_ENTRIES_THIS_DISK] = endrec[zipfile._ECD_ENTRIES_TOTAL]
        return endrec

    zipfile._EndRecData64 = patched_endrec64
    zipfile._EndRecData = patched_endrec
    zipfile._vrbench_multidisk_patch = True


def _fix_split_zip_offsets(zf, first_part, part_starts):
    """Undo zipfile's normal concat offset for split ZIP local headers."""
    infos = zf.infolist()
    if not infos:
        return
    prefix_size = 0
    with open(first_part, 'rb') as f:
        if f.read(4) == b'PK\x07\x08':
            prefix_size = 4
    concat = min(info.header_offset for info in infos) - prefix_size
    if concat <= 0:
        return
    for info in infos:
        info.header_offset -= concat
        if info.volume:
            info.header_offset += part_starts[info.volume]

    end_offset = zf.start_dir
    for info in reversed(sorted(infos, key=lambda item: item.header_offset)):
        info._end_offset = end_offset
        end_offset = info.header_offset


def _file_crc32(path):
    crc = 0
    with open(path, 'rb') as f:
        while True:
            block = f.read(16 * 1024 * 1024)
            if not block:
                break
            crc = zlib.crc32(block, crc)
    return crc & 0xffffffff


def _safe_zip_target(root, member_name):
    root = Path(root).resolve()
    target = (root / member_name).resolve()
    if osp.commonpath([root, target]) != str(root):
        raise zipfile.BadZipFile(f'Archive member escapes extraction root: {member_name}')
    return target


def _load_extract_state(path, archive_parts):
    signature = [
        {'name': part.name, 'size': part.stat().st_size}
        for part in archive_parts
    ]
    state = {'archive_parts': signature, 'completed': {}}
    if path.exists():
        try:
            with path.open('r', encoding='utf-8') as f:
                saved = json.load(f)
            if saved.get('archive_parts') == signature and isinstance(saved.get('completed'), dict):
                state = saved
        except (OSError, ValueError):
            pass
    return state


def _save_extract_state(path, state):
    tmp_path = path.with_name(path.name + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as f:
        json.dump(state, f)
    os.replace(tmp_path, path)


def _dict_to_text(question, options):
    option_prompt = '\n'.join(f'{k}: {v}' for k, v in options.items())
    return f'Question: {question}\nOptions:\n{option_prompt}'


def _prepare_qa_text_input(video_summary, qa_dict, prompt_name):
    prompts = {'mcq': MCQ_COT_PROMPT, 'mcq_think': MCQ_THINK_PROMPT}
    prompt = prompts[prompt_name]
    return prompt.substitute(
        multiple_choice_question=_dict_to_text(qa_dict['question'], qa_dict['options']),
        video_summary=video_summary,
    )


def _format_process(procedure):
    procedure = _loads(procedure, procedure)
    if isinstance(procedure, str):
        return procedure
    sorted_steps = sorted(procedure.items(), key=lambda x: int(x[0]))
    return '\n'.join([f'<Step {num}> {desc}' for num, desc in sorted_steps])


def _middle_frame_indices(total_frames, num_samples):
    if num_samples >= total_frames:
        return list(range(total_frames))
    intervals = np.linspace(0, total_frames, num=num_samples + 1, dtype=int)
    indices = []
    for i in range(num_samples):
        start = intervals[i]
        end = intervals[i + 1]
        indices.append((start + end) // 2)
    return indices


def _normalise_video_relpath(video_path):
    video_path = video_path.replace('\\', '/')
    if '/v001/' in video_path:
        video_path = video_path.replace('/v001/', '/')
    if video_path.startswith('VRBench/videos/'):
        video_path = video_path.replace('VRBench/videos/', 'videos/')
    return video_path


def _extract_mcq_answer(response_text):
    if not response_text:
        return None

    period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
    comma_strip = re.compile(r"(\d)(\,)(\d)")
    punct = [
        ";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\",
        "_", "-", ">", "<", "@", "`", ",", "?", "!"
    ]

    def process_punctuation(text):
        output = text
        for p in punct:
            if (p + " " in text or " " + p in text) or (re.search(comma_strip, text) is not None):
                output = output.replace(p, "")
            else:
                output = output.replace(p, " ")
        output = period_strip.sub("", output, re.UNICODE)
        return output

    boxed_match = re.search(r'\\boxed\{([A-E])\}', response_text, re.IGNORECASE)
    if boxed_match:
        return boxed_match.group(1).upper()
    answer_match = re.search(r'<Answer>\s*([A-E])', response_text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()

    option_format_matches = re.findall(
        r'^([A-E])\.\s*(.+)$',
        response_text.strip(),
        re.IGNORECASE | re.MULTILINE,
    )
    if option_format_matches:
        return option_format_matches[-1][0].upper()

    answer_patterns = [
        r'Answer[:\s]+([A-E])(?:\s|$|\.|,)',
        r'answer is[:\s]+([A-E])(?:\s|$|\.|,)',
        r'correct answer[:\s]+([A-E])(?:\s|$|\.|,)',
        r'final answer[:\s]+([A-E])(?:\s|$|\.|,)',
        r'final[:\s]+([A-E])(?:\s|$|\.|,)',
        r'therefore[:\s]+([A-E])(?:\s|$|\.|,)',
        r'conclusion[:\s]+([A-E])(?:\s|$|\.|,)',
    ]
    all_answer_matches = []
    for pattern in answer_patterns:
        all_answer_matches.extend(re.findall(pattern, response_text, re.IGNORECASE))
    if all_answer_matches:
        return all_answer_matches[-1].upper()

    option_patterns = [
        r'([A-E])\.',
        r'([A-E])\)',
        r'([A-E]):',
    ]
    for pattern in option_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    space_pattern = r'(?:^|\n|\s)([A-E])\s+(?=[A-Z]|$|\n)'
    matches = re.findall(space_pattern, response_text, re.IGNORECASE | re.MULTILINE)
    if matches:
        return matches[-1].upper()

    processed_text = response_text.replace("\n", " ").replace("\t", " ").strip()
    processed_text = process_punctuation(processed_text)
    processed_text = processed_text.strip("'").strip('"').strip(")").strip("(").strip().lower()
    processed_letters = re.findall(r'\b([A-E])\b', processed_text, re.IGNORECASE)
    if processed_letters:
        return processed_letters[-1].upper()

    choices = re.findall(r'\b([A-E])\b', response_text)
    if choices:
        end_portion = response_text.lower()[-50:]
        end_choices = re.findall(r'\b([A-E])\b', end_portion, re.IGNORECASE)
        if end_choices:
            return end_choices[-1].upper()
        return choices[-1].upper()
    return None


def _extract_first_number(value):
    match = re.search(r'[-+]?(?:\d+\.?\d*|\.\d+)', str(value or ''))
    return float(match.group(0)) if match else 0.0


def _extract_tag_text(text, tag):
    match = re.search(rf'<{tag}>(.*?)</{tag}>', str(text or ''), re.DOTALL)
    return match.group(1).strip() if match else ''


def _parse_judge_response(question_type, eval_response, separate=False):
    eval_response = str(eval_response or '')
    if FAIL_MSG in eval_response:
        return '', eval_response
    if not separate:
        rate = _extract_tag_text(eval_response, 'rate')
        reason = _extract_tag_text(eval_response, 'reason')
        return rate, reason

    question_type = str(question_type or '').strip()
    if question_type in UNIQUE_TYPES:
        step_score = _extract_first_number(_extract_tag_text(eval_response, 'step_matching'))
        logical_score = _extract_first_number(_extract_tag_text(eval_response, 'logical_consistency'))
        factual_score = _extract_first_number(_extract_tag_text(eval_response, 'factual_accuracy'))
        clarity_score = _extract_first_number(_extract_tag_text(eval_response, 'process_clarity'))
        weighted_score = step_score * 0.4 + logical_score * 0.4 + factual_score * 0.1 + clarity_score * 0.1
        return str(weighted_score), _extract_tag_text(eval_response, 'rationale')
    if question_type in NON_UNIQUE_TYPES:
        relevance_score = _extract_first_number(_extract_tag_text(eval_response, 'relevance'))
        logical_score = _extract_first_number(_extract_tag_text(eval_response, 'logical_consistency'))
        factual_score = _extract_first_number(_extract_tag_text(eval_response, 'factual_accuracy'))
        clarity_score = _extract_first_number(_extract_tag_text(eval_response, 'clarity'))
        weighted_score = relevance_score * 0.4 + logical_score * 0.4 + factual_score * 0.1 + clarity_score * 0.1
        return str(weighted_score), _extract_tag_text(eval_response, 'rationale')
    return '', ''


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _official_combined_score(mcq_score, openqa_score):
    if mcq_score > 0 and openqa_score > 0:
        return (mcq_score + openqa_score) / 2
    if mcq_score > 0:
        return mcq_score
    if openqa_score > 0:
        return openqa_score
    return 0


class VRBenchDataset(VideoBaseDataset):
    TYPE = 'Video-MCQ'
    DEFAULT_JUDGE_MODEL = 'deepseek-v4-flash-beta'
    HF_REPO_ID = 'OpenGVLab/VRBench'

    def __init__(self, dataset='VRBench', pack=False, nframe=0, fps=-1, prompt='mcq'):
        self.prompt = prompt
        self._frame_cache = {}
        self._frame_locks = {}
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['VRBench']

    def prepare_dataset(self, dataset_name='VRBench'):
        data_root = LMUDataRoot()
        dataset_root = osp.join(data_root, 'VRBench')
        os.makedirs(dataset_root, exist_ok=True)
        data_file = osp.join(dataset_root, f'{dataset_name}.tsv')

        if osp.exists(data_file):
            self._ensure_video_symlinks(dataset_root, data_file)
        else:
            src_file = self._source_file(dataset_name, dataset_root)
            self._build_tsv(src_file, data_file)
            self._ensure_video_symlinks(dataset_root, data_file)
        if not self._videos_ready(dataset_root, data_file):
            self._download_video_archives(dataset_root)
            self._extract_video_archives(dataset_root)
            self._ensure_video_symlinks(dataset_root, data_file)
        if not self._videos_ready(dataset_root, data_file):
            raise FileNotFoundError(
                f'VRBench videos are incomplete under {dataset_root}. '
                'Expected every video_path in the TSV to exist after download/extraction.'
            )
        return dict(data_file=data_file, root=dataset_root)

    def _source_file(self, dataset_name, dataset_root):
        _ensure_hf_cache_env(dataset_root)
        try:
            from huggingface_hub import hf_hub_download
        except Exception as e:
            raise ImportError('huggingface_hub is required to download VRBench.') from e
        return hf_hub_download(
            repo_id=self.HF_REPO_ID,
            repo_type='dataset',
            filename='VRBench_eval.jsonl',
            local_dir=dataset_root,
        )

    def _download_video_archives(self, dataset_root):
        _ensure_hf_cache_env(dataset_root)
        try:
            from huggingface_hub import hf_hub_download, hf_hub_url
            from huggingface_hub.file_download import (_is_same_or_hub_host, get_hf_file_metadata,
                                                       http_get)
            from huggingface_hub.utils import build_hf_headers
        except Exception as e:
            raise ImportError('huggingface_hub is required to download VRBench videos.') from e

        archive_count = len(VIDEO_ARCHIVE_FILES)
        backend = os.environ.get('VRBENCH_DOWNLOAD_BACKEND', 'http_range').lower()
        file_workers = _env_int('VRBENCH_DOWNLOAD_FILE_WORKERS', 1)

        def download_one(archive_idx, filename):
            dst_path = Path(dataset_root) / filename
            part_path = dst_path.with_name(dst_path.name + '.part')
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            if backend in {'hf', 'hf_hub', 'xet'}:
                if dst_path.exists():
                    print(
                        f'VRBench archive {archive_idx}/{archive_count} already present: {filename}',
                        flush=True,
                    )
                    return
                print(
                    f'Downloading VRBench archive {archive_idx}/{archive_count} via hf_hub_download: '
                    f'{filename}',
                    flush=True,
                )
                hf_hub_download(
                    repo_id=self.HF_REPO_ID,
                    repo_type='dataset',
                    filename=filename,
                    local_dir=dataset_root,
                )
                return

            url = hf_hub_url(self.HF_REPO_ID, filename, repo_type='dataset')
            headers = build_hf_headers(token=None)
            headers['Accept-Encoding'] = 'identity'
            metadata = get_hf_file_metadata(
                url,
                headers=headers,
                token=None,
                timeout=30,
                retry_on_errors=True,
            )
            expected_size = metadata.size
            download_url = url if metadata.xet_file_data is not None else metadata.location
            download_headers = dict(headers)
            if metadata.xet_file_data is None and not _is_same_or_hub_host(url, metadata.location):
                download_headers.pop('authorization', None)

            if dst_path.exists():
                if expected_size is None or dst_path.stat().st_size == expected_size:
                    print(
                        f'VRBench archive {archive_idx}/{archive_count} already present: {filename}',
                        flush=True,
                    )
                    return
                raise OSError(
                    f'Existing VRBench archive has unexpected size: {dst_path} '
                    f'({dst_path.stat().st_size} != {expected_size}).'
                )

            part_size = part_path.stat().st_size if part_path.exists() else 0
            if expected_size is not None and part_size > expected_size:
                raise OSError(
                    f'Partial VRBench archive is larger than expected: {part_path} '
                    f'({part_size} > {expected_size}).'
                )
            if expected_size is None:
                print(
                    f'Downloading VRBench archive {archive_idx}/{archive_count}: '
                    f'{filename} ({part_size}/unknown bytes)',
                    flush=True,
                )
                with part_path.open('ab' if part_size else 'wb') as f:
                    http_get(
                        url=download_url,
                        temp_file=f,
                        resume_size=part_size,
                        headers=download_headers,
                        expected_size=expected_size,
                        displayed_filename=filename,
                    )
                os.replace(part_path, dst_path)
            else:
                print(
                    f'Downloading VRBench archive {archive_idx}/{archive_count}: '
                    f'{filename} (expected {expected_size} bytes)',
                    flush=True,
                )
                _download_with_http_ranges(
                    download_url,
                    download_headers,
                    dst_path,
                    filename,
                    expected_size,
                )

        if file_workers <= 1 or backend in {'hf', 'hf_hub', 'xet'}:
            for archive_idx, filename in enumerate(VIDEO_ARCHIVE_FILES, start=1):
                download_one(archive_idx, filename)
            return

        print(
            f'Downloading VRBench archives with file-level workers={file_workers}',
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=min(file_workers, archive_count)) as executor:
            futures = {
                executor.submit(download_one, archive_idx, filename): filename
                for archive_idx, filename in enumerate(VIDEO_ARCHIVE_FILES, start=1)
            }
            for future in as_completed(futures):
                future.result()

    def _build_tsv(self, src_file, data_file):
        if osp.exists(data_file) and osp.getmtime(data_file) >= osp.getmtime(src_file):
            return
        records = []
        with open(src_file, 'r', encoding='utf-8') as f:
            for video_idx, line in enumerate(f):
                item = json.loads(line)
                video_id = item['video_id']
                video_rel = _normalise_video_relpath(item['video_path'])
                video_stem = osp.splitext(video_rel)[0]
                for qa_id, qa in item['mcq'].items():
                    records.append({
                        'index': len(records),
                        'id': f'{video_id}_{qa_id}',
                        'video': video_stem,
                        'video_id': video_id,
                        'qa_id': qa_id,
                        'video_path': video_rel,
                        'source_video_path': item['video_path'],
                        'video_summary': item.get('video_summary', ''),
                        'video_read_type': item.get('video_read_type', 'decord'),
                        'question': qa['question'],
                        'options': json.dumps(qa['options'], ensure_ascii=False),
                        'answer': qa.get('answer', ''),
                        'original_question': qa.get('original_question', qa['question']),
                        'original_answer': qa.get('original_answer', ''),
                        'reasoning_process': json.dumps(qa.get('reasoning_process', {}), ensure_ascii=False),
                        'reasoning_type': qa.get('reasoning_type', ''),
                        'video_index': video_idx,
                    })
        pd.DataFrame(records).to_csv(data_file, sep='\t', index=False)

    def _extract_video_archives(self, dataset_root):
        archive_dir = Path(dataset_root) / 'v001_360p_zips'
        archives = sorted(archive_dir.glob('*.zip'))
        if not archives:
            warnings.warn(
                f'VRBench video archives were not found under {archive_dir}. '
                'Only metadata is available until the archives are downloaded.'
            )
            return
        lock_path = osp.join(dataset_root, '.vrbench_extract.lock')
        with portalocker.Lock(lock_path, 'w', timeout=30):
            for archive in archives:
                split_parts = sorted(
                    archive_dir.glob(f'{archive.stem}.z[0-9][0-9]'),
                    key=lambda p: p.suffix,
                )
                if not split_parts:
                    subprocess.run(['unzip', '-o', str(archive), '-d', dataset_root], check=True)
                    archive.unlink()
                    continue

                archive_parts = [*split_parts, archive]
                state_path = archive_dir / f'.{archive.stem}.extract.json'
                state = _load_extract_state(state_path, archive_parts)
                completed = state['completed']
                _patch_zipfile_multidisk_for_concat()
                with _ConcatBinaryFile(archive_parts) as fp:
                    with zipfile.ZipFile(fp) as zf:
                        _fix_split_zip_offsets(zf, split_parts[0], fp.starts)
                        members = [info for info in zf.infolist() if not info.is_dir()]
                        total = len(members)
                        for index, info in enumerate(members, start=1):
                            target = _safe_zip_target(dataset_root, info.filename)
                            identity = f'{info.file_size}:{info.CRC}'
                            already_recorded = completed.get(info.filename) == identity
                            complete = target.is_file() and target.stat().st_size == info.file_size
                            if complete and not already_recorded:
                                complete = _file_crc32(target) == info.CRC
                            if complete:
                                if not already_recorded:
                                    completed[info.filename] = identity
                                    _save_extract_state(state_path, state)
                                print(
                                    f'VRBench extract {index}/{total} skip complete: {info.filename}',
                                    flush=True,
                                )
                                continue

                            target.parent.mkdir(parents=True, exist_ok=True)
                            part_path = target.with_name(target.name + '.vrbench.part')
                            print(f'VRBench extract {index}/{total}: {info.filename}', flush=True)
                            with zf.open(info, 'r') as src, part_path.open('wb') as dst:
                                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
                            if part_path.stat().st_size != info.file_size:
                                raise OSError(
                                    f'Extracted size mismatch for {info.filename}: '
                                    f'{part_path.stat().st_size} != {info.file_size}'
                                )
                            os.replace(part_path, target)
                            completed[info.filename] = identity
                            _save_extract_state(state_path, state)

                state_path.unlink(missing_ok=True)
                for part in archive_parts:
                    part.unlink()

    def _videos_ready(self, dataset_root, data_file):
        if not osp.exists(data_file):
            return False
        data = pd.read_csv(data_file, sep='\t')
        if 'video_path' not in data:
            return False
        for video_path in data['video_path'].dropna().unique():
            if not osp.exists(osp.join(dataset_root, video_path)):
                return False
        return True

    def _ensure_video_symlinks(self, dataset_root, data_file):
        data = pd.read_csv(data_file, sep='\t')
        target_dir = osp.join(dataset_root, 'videos')
        os.makedirs(target_dir, exist_ok=True)
        for _, row in data.drop_duplicates('video_id').iterrows():
            dst = osp.join(dataset_root, row['video_path'])
            if osp.exists(dst):
                continue
            if osp.lexists(dst):
                os.unlink(dst)
            basename = osp.basename(row['video_path'])
            candidates = [
                osp.join(dataset_root, 'VRBench', 'videos', 'v001', basename),
                osp.join(dataset_root, 'videos', 'v001', basename),
                osp.join(dataset_root, 'v001', basename),
                osp.join(dataset_root, 'v001_360p', basename),
            ]
            src = next((p for p in candidates if osp.exists(p)), None)
            if src is None:
                continue
            os.makedirs(osp.dirname(dst), exist_ok=True)
            rel_src = osp.relpath(src, osp.dirname(dst))
            os.symlink(rel_src, dst)

    def _video_path(self, line):
        video_path = osp.join(self.data_root, line['video_path'])
        if not osp.exists(video_path):
            raise FileNotFoundError(
                f'VRBench video file is missing: {video_path}. '
                'Prepare or download/extract VRBench videos under LMUDataRoot()/VRBench.'
            )
        return video_path

    def _frame_paths(self, video_id, indices, suffix):
        frame_root = osp.join(self.frame_root, video_id)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, f'frame-{idx}-of-{suffix}.jpg') for idx in indices]

    def _cached_frame_paths_from_disk(self, line, num_frames, cache_key):
        if num_frames <= 0 or os.environ.get('VRBENCH_FRAME_HASH'):
            return None
        frame_root = osp.join(self.frame_root, line['video_id'])
        if not osp.isdir(frame_root):
            return None
        pattern = re.compile(rf'^frame-(\d+)-of-{re.escape(str(num_frames))}\.jpg$')
        indexed_paths = []
        for name in os.listdir(frame_root):
            match = pattern.match(name)
            if match:
                indexed_paths.append((int(match.group(1)), osp.join(frame_root, name)))
        if len(indexed_paths) != num_frames:
            return None
        indexed_paths.sort(key=lambda item: item[0])
        indices = [idx for idx, _ in indexed_paths]
        frame_paths = [path for _, path in indexed_paths]
        if not np.all([osp.exists(path) for path in frame_paths]):
            return None
        result = (frame_paths, indices, {})
        self._frame_cache[cache_key] = result
        return result

    def _read_indices_with_decord(self, video_path, num_frames):
        import decord
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
        total = len(vr)
        if total == 0:
            raise ValueError(f'Empty video: {video_path}')
        if num_frames == -1:
            fps = vr.get_avg_fps()
            interval = max(1, int(fps))
            indices = list(range(0, total, interval))
        else:
            indices = _middle_frame_indices(total, num_frames)
        frames = []
        for idx in indices:
            frame = vr[idx].asnumpy()
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        return frames, indices, {'fps': vr.get_avg_fps(), 'n_frames': total}

    def _read_indices_with_av(self, video_path, num_frames):
        import av
        container = av.open(video_path)
        stream = container.streams.video[0]
        total = stream.frames
        if num_frames == -1:
            fps = float(stream.average_rate)
            interval = max(1, int(fps))
            indices = list(range(0, total, interval))
        else:
            indices = _middle_frame_indices(total, num_frames)
        index_set = set(indices)
        frames = []
        for idx, frame in enumerate(container.decode(stream)):
            if idx > max(index_set):
                break
            if idx in index_set:
                arr = frame.reformat(format='rgb24').to_ndarray()
                arr = cv2.resize(arr, (224, 224))
                frames.append(arr)
        return frames, indices, {'fps': float(stream.average_rate), 'n_frames': total}

    def save_video_frames(self, line):
        video_path = self._video_path(line)
        video_read_type = line.get('video_read_type', 'decord')
        num_frames = self.nframe if self.nframe > 0 else -1
        cache_key = (video_path, video_read_type, num_frames)
        cached = self._frame_cache.get(cache_key)
        if cached is not None and np.all([osp.exists(p) for p in cached[0]]):
            return cached
        cached = self._cached_frame_paths_from_disk(line, num_frames, cache_key)
        if cached is not None:
            return cached
        frame_lock = self._frame_locks.setdefault(cache_key, threading.Lock())
        with frame_lock:
            cached = self._frame_cache.get(cache_key)
            if cached is not None and np.all([osp.exists(p) for p in cached[0]]):
                return cached
            cached = self._cached_frame_paths_from_disk(line, num_frames, cache_key)
            if cached is not None:
                return cached
            return self._save_video_frames_uncached(
                line, video_path, video_read_type, num_frames, cache_key)

    def _save_video_frames_uncached(self, line, video_path, video_read_type, num_frames, cache_key):
        if video_read_type == 'av':
            try:
                frames, indices, video_info = self._read_indices_with_av(video_path, num_frames)
            except ModuleNotFoundError:
                frames, indices, video_info = self._read_indices_with_decord(video_path, num_frames)
        else:
            try:
                frames, indices, video_info = self._read_indices_with_decord(video_path, num_frames)
            except Exception:
                frames, indices, video_info = self._read_indices_with_av(video_path, num_frames)
        video_info['frame_sha256'] = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]
        frame_paths = self._frame_paths(line['video_id'], indices, num_frames)
        if np.all([osp.exists(p) for p in frame_paths]):
            result = (frame_paths, indices, video_info)
            self._frame_cache[cache_key] = result
            return result
        lock_path = osp.join(self.frame_root, f"{line['video_id']}.lock")
        with portalocker.Lock(lock_path, 'w', timeout=30):
            if np.all([osp.exists(p) for p in frame_paths]):
                result = (frame_paths, indices, video_info)
                self._frame_cache[cache_key] = result
                return result
            for frame, path in zip(frames, frame_paths):
                if not osp.exists(path):
                    Image.fromarray(frame).save(path)
        result = (frame_paths, indices, video_info)
        self._frame_cache[cache_key] = result
        return result

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        qa = {
            'question': line['question'],
            'options': _loads(line['options'], {}),
        }
        qa_text_prompt = _prepare_qa_text_input(line['video_summary'], qa, self.prompt)
        if video_llm:
            return [
                dict(type='video', value=self._video_path(line)),
                dict(type='text', value=qa_text_prompt),
            ]
        frames, indices, video_info = self.save_video_frames(line)
        frame_hashes = video_info.get('frame_sha256', [])
        message = []
        for pos, (frame_path, frame_index) in enumerate(zip(frames, indices)):
            item = dict(type='image', value=frame_path, frame_index=int(frame_index))
            if os.environ.get('VRBENCH_FRAME_HASH') and pos < len(frame_hashes):
                item['frame_sha256'] = frame_hashes[pos]
            message.append(item)
        message.append(dict(type='text', value=qa_text_prompt))
        return message

    @staticmethod
    def _judge_messages(line, response, separate=False):
        qtype = str(line['reasoning_type']).strip()
        question = line['question']
        answer = line['original_answer']
        procedure = _format_process(line['reasoning_process'])
        response = str(response).strip('Assistant:')
        if qtype in UNIQUE_TYPES:
            system_prompt = UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT_SEPARATE if separate else UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT
            # Match the official VRBench evaluator call order: procedure is passed as answer, answer as procedure.
            user_prompt = UNIQUE_ANSWER_EVAL_HUMAN_PROMPT_TEMPLATE.format(
                question=question,
                response=response,
                answer=procedure,
                procedure=answer,
            )
        elif qtype in NON_UNIQUE_TYPES:
            system_prompt = (
                NON_UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT_SEPARATE if separate
                else NON_UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT
            )
            user_prompt = NON_UNIQUE_ANSWER_EVAL_HUMAN_PROMPT_TEMPLATE.format(
                video_summary=line['video_summary'][:10000],
                question=question,
                response=response,
                answer=procedure,
                procedure=answer,
            )
        elif qtype in {'Counting Porblems', 'Counting Problems', 'Event Summarization'}:
            return []
        else:
            warnings.warn(f'Unknown VRBench reasoning type {qtype} for {line.get("id", "")}.')
            return []
        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv']
        data = load(eval_file)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        judge_model = judge_kwargs.pop('model', cls.DEFAULT_JUDGE_MODEL)
        nproc = int(judge_kwargs.pop('nproc', 1) or 1)
        separate = bool(judge_kwargs.pop('separate', False))
        judge_file = get_intermediate_file_path(eval_file, f'_{judge_model}_vrbench_judge', 'jsonl')
        score_file = get_intermediate_file_path(eval_file, f'_{judge_model}_score', 'json')
        cache_file = get_intermediate_file_path(eval_file, f'_{judge_model}_vrbench_judge_cache', 'pkl')

        eval_rows = []
        pending = []
        judge_cache = {}
        if osp.exists(cache_file):
            try:
                judge_cache = load(cache_file)
                if not isinstance(judge_cache, dict):
                    judge_cache = {}
                judge_cache = {k: v for k, v in judge_cache.items() if FAIL_MSG not in str(v)}
            except Exception as err:
                warnings.warn(f'Failed to load VRBench judge cache {cache_file}: {err}')
                judge_cache = {}

        mcq_total = 0
        mcq_correct = 0
        mcq_category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        for row_idx, (_, row) in enumerate(data.iterrows()):
            row = row.to_dict()
            question_id = row.get('id') or f"{row['video_id']}_{row['qa_id']}"
            question_id = str(question_id)
            prediction = str(row.get('prediction', ''))
            pred_answer = _extract_mcq_answer(prediction)
            answer = row.get('answer')
            question_type = str(row.get('reasoning_type', '')).strip()
            mcq_total += 1
            mcq_category_stats[question_type]['total'] += 1
            if pred_answer == answer:
                mcq_correct += 1
                mcq_category_stats[question_type]['correct'] += 1

            messages = cls._judge_messages(row, prediction, separate=separate)
            eval_response = judge_cache.get(question_id, None)
            if eval_response is None and messages:
                pending.append((row_idx, question_id, messages))
            elif eval_response is None:
                eval_response = ''
            eval_rows.append({
                'id': question_id,
                'question': row.get('question', ''),
                'answer': row.get('original_answer', ''),
                'procedure': _format_process(row.get('reasoning_process', '')),
                'steps_and_answer': prediction,
                'type': row.get('reasoning_type', ''),
                'eval_response': eval_response,
            })

        if pending:
            from .utils import build_judge
            thread_state = threading.local()

            def get_model():
                model = getattr(thread_state, 'judge_model', None)
                if model is None:
                    model = build_judge(model=judge_model, **judge_kwargs)
                    thread_state.judge_model = model
                return model

            def judge_one(messages):
                try:
                    eval_response = get_model().chat(messages)
                except Exception as err:
                    return f'{FAIL_MSG} {type(err).__name__}: {err}'
                if eval_response is None or eval_response == '':
                    return FAIL_MSG
                return str(eval_response)

            pending_payloads = [dict(messages=messages) for _, _, messages in pending]
            pending_keys = [question_id for _, question_id, _ in pending]
            new_results = track_progress_rich(
                judge_one,
                pending_payloads,
                nproc=nproc,
                chunksize=nproc,
                keys=pending_keys,
                save=cache_file,
            )
            for (row_idx, question_id, _), eval_response in zip(pending, new_results):
                eval_rows[row_idx]['eval_response'] = eval_response
                judge_cache[question_id] = eval_response

            if osp.exists(cache_file):
                try:
                    cached_results = load(cache_file)
                    if isinstance(cached_results, dict):
                        judge_cache.update(cached_results)
                except Exception as err:
                    warnings.warn(f'Failed to reload VRBench judge cache {cache_file}: {err}')
            dump(judge_cache, cache_file)

        openqa_total = 0
        openqa_score_sum = 0.0
        openqa_category_scores = defaultdict(list)
        for row in eval_rows:
            eval_response = str(row.get('eval_response', '') or '')
            rate, reason = _parse_judge_response(row.get('type', ''), eval_response, separate=separate)
            row['rate'] = rate
            row['reason'] = reason
            score = _safe_float(rate)
            openqa_total += 1
            openqa_score_sum += score
            openqa_category_scores[str(row.get('type', '')).strip()].append(score)

        dump(eval_rows, judge_file)
        mcq_o = (mcq_correct / mcq_total * 100) if mcq_total else 0
        oe_p = (openqa_score_sum / openqa_total * 10) if openqa_total else 0
        overall = (mcq_o + oe_p) / 2

        category_scores = {}
        all_categories = set(mcq_category_stats.keys()) | set(openqa_category_scores.keys())
        for category in sorted(all_categories):
            mcq_stats = mcq_category_stats.get(category, {'correct': 0, 'total': 0})
            category_mcq = (
                mcq_stats['correct'] / mcq_stats['total'] * 100
                if mcq_stats['total'] else 0
            )
            openqa_scores = openqa_category_scores.get(category, [])
            category_openqa = (
                sum(openqa_scores) / len(openqa_scores) * 10
                if openqa_scores else 0
            )
            category_scores[category] = {
                'mcq_accuracy': round(category_mcq, 2),
                'openqa_average': round(category_openqa, 2),
                'combined_score': round(_official_combined_score(category_mcq, category_openqa), 2),
                'mcq_stats': f"{mcq_stats['correct']}/{mcq_stats['total']}",
                'openqa_stats': f"{len(openqa_scores)} questions",
            }

        result = {
            'Overall': round(overall, 2),
            'MCQ-O': round(mcq_o, 2),
            'OE-P': round(oe_p, 2),
            'overall': round(overall, 2),
            'mcq_accuracy': round(mcq_o, 2),
            'openqa_average': round(oe_p, 2),
            'mcq_stats': f'{mcq_correct}/{mcq_total}',
            'openqa_stats': f'{openqa_total} questions',
            'category_scores': category_scores,
            'judge_file': judge_file,
        }
        dump(result, score_file)
        return result
