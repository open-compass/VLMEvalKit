import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from src.metrics.cdm.modules.texlive_env import build_tex_env, resolve_cjk_font_family, resolve_tex_binary


_VERSION_LINE_PATTERNS = [
    re.compile(r'\\def\\fileversion\{([^}]+)\}'),
    re.compile(r'\\Provides(?:Package|File)\{[^}]+\}\[([^\]]+)\]'),
    re.compile(r'^\s*%+\s*Version\s+(.+)$', re.I),
]

_TRACKED_SYSTEM_PACKAGES = [
    'ghostscript',
    'imagemagick',
    'fontconfig',
    'python3',
    'python3-pip',
    'python3-setuptools',
    'python3-wheel',
    'texlive-base',
    'texlive-binaries',
    'texlive-lang-chinese',
    'texlive-lang-cjk',
    'texlive-latex-base',
    'texlive-latex-extra',
    'texlive-fonts-recommended',
    'texlive-plain-generic',
    'latexml',
]


def _safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _safe_get(mapping, *path, default=None):
    current = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _normalize_img_name(img_id):
    img_id = str(img_id or '')
    if not img_id:
        return ''
    if img_id.endswith('.jpg') or img_id.endswith('.png'):
        return img_id
    if '_' in img_id:
        return '_'.join(img_id.split('_')[:-1])
    return img_id


def build_metric_page_denominators(samples, gt_page_names=None, expected_metrics=None):
    metric_names = list(expected_metrics or [])
    if 'TEDS' in metric_names and 'TEDS_structure_only' not in metric_names:
        metric_names.append('TEDS_structure_only')

    sample_pages = {
        _normalize_img_name(sample.get('img_id'))
        for sample in (samples or [])
        if sample.get('metric') and _normalize_img_name(sample.get('img_id'))
    }
    page_count = len(set(gt_page_names or [])) if gt_page_names else len(sample_pages)

    return {
        metric_name: {
            'ALL': page_count,
        }
        for metric_name in metric_names
    }


def build_notebook_metric_summary(result_all, page_denominators):
    """Mirror tools/generate_result_tables.ipynb cell 2."""
    summary = {
        'source': 'tools/generate_result_tables.ipynb#cell-2',
        'metrics': {},
    }

    text_edit = _safe_float(_safe_get(result_all, 'text_block', 'all', 'Edit_dist', 'ALL_page_avg'))
    display_formula_cdm = _safe_float(_safe_get(result_all, 'display_formula', 'page', 'CDM', 'ALL'))
    table_teds = _safe_float(_safe_get(result_all, 'table', 'page', 'TEDS', 'ALL'))
    table_teds_structure_only = _safe_float(_safe_get(result_all, 'table', 'page', 'TEDS_structure_only', 'ALL'))
    reading_order_edit = _safe_float(_safe_get(result_all, 'reading_order', 'all', 'Edit_dist', 'ALL_page_avg'))

    summary['metrics']['text_block_Edit_dist'] = {
        'raw': text_edit,
        'page_denominator': _safe_get(page_denominators, 'text_block', 'Edit_dist', 'ALL'),
        'notebook_value': text_edit,
    }
    summary['metrics']['display_formula_CDM'] = {
        'raw': display_formula_cdm,
        'page_denominator': _safe_get(page_denominators, 'display_formula', 'CDM', 'ALL'),
        'notebook_value': None if display_formula_cdm is None else display_formula_cdm * 100.0,
    }
    summary['metrics']['table_TEDS'] = {
        'raw': table_teds,
        'page_denominator': _safe_get(page_denominators, 'table', 'TEDS', 'ALL'),
        'notebook_value': None if table_teds is None else table_teds * 100.0,
    }
    summary['metrics']['table_TEDS_structure_only'] = {
        'raw': table_teds_structure_only,
        'page_denominator': _safe_get(page_denominators, 'table', 'TEDS_structure_only', 'ALL'),
        'notebook_value': None if table_teds_structure_only is None else table_teds_structure_only * 100.0,
    }
    summary['metrics']['reading_order_Edit_dist'] = {
        'raw': reading_order_edit,
        'page_denominator': _safe_get(page_denominators, 'reading_order', 'Edit_dist', 'ALL'),
        'notebook_value': reading_order_edit,
    }

    if text_edit is not None and display_formula_cdm is not None and table_teds is not None:
        summary['overall_notebook'] = ((1.0 - text_edit) * 100.0 + display_formula_cdm * 100.0 + table_teds * 100.0) / 3.0
    else:
        summary['overall_notebook'] = None

    return summary


def _sort_page_names(page_names):
    return sorted({str(page_name) for page_name in (page_names or []) if str(page_name)})


def _sort_case_records(case_records):
    normalized = []
    for case in case_records or []:
        if isinstance(case, dict):
            normalized.append({
                key: value
                for key, value in case.items()
                if value not in (None, '', [])
            })
        elif case not in (None, ''):
            normalized.append({'case_name': str(case)})
    return sorted(
        normalized,
        key=lambda item: (
            str(item.get('img_id', '')),
            str(item.get('gt_idx', '')),
            str(item.get('pred_idx', '')),
            str(item.get('reason', '')),
            str(item.get('case_name', '')),
        ),
    )


def build_stage_execution_summary(result_all):
    summary = {}

    match_debug = result_all.get('match_debug') if isinstance(result_all, dict) else None
    if isinstance(match_debug, dict):
        fallback_pages = match_debug.get('text_match_fallback_pages', {})
        fallback_counts = match_debug.get('text_match_fallback_counts', {})
        summary['page_match'] = {
            'workers': match_debug.get('workers', match_debug.get('match_workers')),
            'page_count': match_debug.get('page_count'),
            'quick_match_truncated_timeout_sec': match_debug.get('quick_match_truncated_timeout_sec'),
            'match_timeout_sec': match_debug.get('match_timeout_sec'),
            'fallbacks': {
                reason: {
                    'count': int(fallback_counts.get(reason, len(_sort_page_names(pages)))),
                    'cases': _sort_page_names(pages),
                }
                for reason, pages in fallback_pages.items()
            },
        }

    metric_summary = {}
    if isinstance(result_all, dict):
        for element, payload in result_all.items():
            if element == 'match_debug' or not isinstance(payload, dict):
                continue
            metric_debug = payload.get('metric_debug')
            if not isinstance(metric_debug, dict):
                continue
            element_summary = {}
            for metric_name, debug in metric_debug.items():
                if not isinstance(debug, dict):
                    continue
                element_summary[metric_name] = {
                    'workers': debug.get('workers'),
                    'timeout_sec': debug.get('timeout_sec'),
                    'sample_count': debug.get('sample_count'),
                    'timeout_case_count': int(debug.get('timeout_case_count', len(debug.get('timeout_cases', [])))),
                    'timeout_cases': _sort_case_records(debug.get('timeout_cases', [])),
                    'error_case_count': int(debug.get('error_case_count', len(debug.get('error_cases', [])))),
                    'error_cases': _sort_case_records(debug.get('error_cases', [])),
                    'exception_case_count': int(debug.get('exception_case_count', len(debug.get('exception_cases', [])))),
                    'exception_cases': _sort_case_records(debug.get('exception_cases', [])),
                }
            if element_summary:
                metric_summary[element] = element_summary
    if metric_summary:
        summary['metrics'] = metric_summary

    return summary


def _read_os_release():
    path = Path('/etc/os-release')
    info = {}
    if not path.exists():
        return info
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        info[key] = value.strip().strip('"')
    return info


def _load_project_dependency_names(pyproject_path):
    dependencies = []
    if not pyproject_path.exists():
        return dependencies

    in_project = False
    in_dependencies = False
    for raw_line in pyproject_path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = raw_line.strip()
        if line == '[project]':
            in_project = True
            continue
        if in_project and line.startswith('[') and line != '[project]':
            break
        if not in_project:
            continue
        if line.startswith('dependencies'):
            in_dependencies = True
            continue
        if in_dependencies:
            if line.startswith(']'):
                break
            match = re.search(r'"([^"]+)"', line)
            if match:
                requirement = match.group(1)
                name_match = re.match(r'^\s*([A-Za-z0-9_.-]+)', requirement)
                if name_match:
                    dependencies.append(name_match.group(1))
    return dependencies


def _distribution_version(distribution_name):
    candidates = [
        distribution_name,
        distribution_name.replace('-', '_'),
        distribution_name.replace('_', '-'),
        distribution_name.lower(),
        distribution_name.lower().replace('-', '_'),
        distribution_name.lower().replace('_', '-'),
    ]
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return 'unavailable'


def _collect_dependency_versions(repo_root):
    pyproject_path = repo_root / 'pyproject.toml'
    dependency_names = _load_project_dependency_names(pyproject_path)
    versions = {}
    for dependency_name in dependency_names:
        versions[dependency_name] = _distribution_version(dependency_name)
    for dependency_name in ['pip', 'setuptools', 'wheel']:
        versions.setdefault(dependency_name, _distribution_version(dependency_name))
    return versions


def _query_dpkg_package(package_name):
    try:
        completed = subprocess.run(
            ['dpkg-query', '-W', f'--showformat=${{Package}}=${{Version}}', package_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return f'unavailable ({type(exc).__name__}: {exc})'

    output = (completed.stdout or '').strip()
    if completed.returncode == 0 and output:
        return output
    if output:
        return f'unavailable ({output})'
    return 'unavailable'


def _query_rpm_package(package_name):
    try:
        completed = subprocess.run(
            ['rpm', '-q', package_name, '--qf', '%{NAME}=%{VERSION}-%{RELEASE}'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return f'unavailable ({type(exc).__name__}: {exc})'

    output = (completed.stdout or '').strip()
    if completed.returncode == 0 and output:
        return output
    if output:
        return f'unavailable ({output})'
    return 'unavailable'


def _query_apk_package(package_name):
    try:
        completed = subprocess.run(
            ['apk', 'info', '-v', package_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return f'unavailable ({type(exc).__name__}: {exc})'

    lines = [line.strip() for line in (completed.stdout or '').splitlines() if line.strip()]
    if completed.returncode == 0 and lines:
        return lines[0]
    if lines:
        return f'unavailable ({lines[0]})'
    return 'unavailable'


def _collect_system_package_versions():
    if shutil.which('dpkg-query'):
        package_manager = 'dpkg-query'
        query_fn = _query_dpkg_package
    elif shutil.which('rpm'):
        package_manager = 'rpm'
        query_fn = _query_rpm_package
    elif shutil.which('apk'):
        package_manager = 'apk'
        query_fn = _query_apk_package
    else:
        return {
            'package_manager': 'unavailable',
            'tracked_packages': {package_name: 'unavailable (no supported package manager found)' for package_name in _TRACKED_SYSTEM_PACKAGES},
        }

    return {
        'package_manager': package_manager,
        'tracked_packages': {
            package_name: query_fn(package_name)
            for package_name in _TRACKED_SYSTEM_PACKAGES
        },
    }


def _run_command(command, timeout=5, env=None):
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
        lines = [line.strip() for line in (completed.stdout or '').splitlines() if line.strip()]
        return {
            'path': shutil.which(command[0]) or command[0],
            'returncode': completed.returncode,
            'output': ' | '.join(lines[:5]) if lines else '',
        }
    except Exception as exc:
        return {
            'path': shutil.which(command[0]) or command[0],
            'returncode': None,
            'output': f'unavailable ({type(exc).__name__}: {exc})',
        }


def _read_tex_file_metadata(path_str):
    raw_path = str(path_str or '').strip()
    if not raw_path:
        return {
            'path': '',
            'version': 'unavailable',
            'date': '',
            'header': '',
            'status': 'missing',
        }

    path = Path(raw_path)
    if not path.exists():
        return {
            'path': raw_path,
            'version': 'unavailable',
            'date': '',
            'header': '',
            'status': 'not_found',
        }
    if path.is_dir():
        return {
            'path': str(path),
            'version': 'unavailable',
            'date': '',
            'header': '',
            'status': 'directory',
        }

    try:
        header_lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()[:80]
    except Exception as exc:
        return {
            'path': str(path),
            'version': 'unavailable',
            'date': '',
            'header': '',
            'status': f'error ({type(exc).__name__})',
            'error': f'{type(exc).__name__}: {exc}',
        }

    version = 'unavailable'
    date = ''
    header = ''
    for line in header_lines:
        stripped = line.strip()
        if not header and stripped:
            header = stripped
        for pattern in _VERSION_LINE_PATTERNS:
            match = pattern.search(stripped)
            if match and version == 'unavailable':
                version = match.group(1).strip()
        if not date:
            date_match = re.search(r'\\def\\filedate\{([^}]+)\}', stripped)
            if date_match:
                date = date_match.group(1).strip()

    return {
        'path': str(path),
        'version': version,
        'date': date,
        'header': header,
        'status': 'ok',
    }


def collect_runtime_environment_report():
    repo_root = Path(__file__).resolve().parents[2]
    tex_env = build_tex_env()
    cjk_font_family = resolve_cjk_font_family()
    cjk_sty_path = _run_command([resolve_tex_binary('kpsewhich'), 'CJK.sty'], env=tex_env).get('output', '').split(' | ')[0]
    cjk_fd_path = _run_command([resolve_tex_binary('kpsewhich'), f'c70{cjk_font_family}.fd'], env=tex_env).get('output', '').split(' | ')[0]

    return {
        'system': {
            'platform': platform.platform(),
            'python_version': sys.version.replace('\n', ' '),
            'python_executable': sys.executable,
            'python_prefix': sys.prefix,
            'conda_env': os.getenv('CONDA_DEFAULT_ENV', ''),
            'conda_prefix': os.getenv('CONDA_PREFIX', ''),
            'libc': ' '.join(part for part in platform.libc_ver() if part),
            'uname': ' '.join(platform.uname()),
            'os_release': _read_os_release(),
        },
        'system_packages': _collect_system_package_versions(),
        'python_packages': _collect_dependency_versions(repo_root),
        'external_tools': {
            'pdflatex': _run_command([resolve_tex_binary('pdflatex'), '--version'], env=tex_env),
            'kpsewhich': _run_command([resolve_tex_binary('kpsewhich'), '--version'], env=tex_env),
            'tlmgr': _run_command(['tlmgr', '--version'], env=tex_env),
            'ghostscript': _run_command(['gs', '--version']),
            'magick': _run_command(['magick', '-version']),
            'convert': _run_command(['convert', '-version']),
            'identify': _run_command(['identify', '-version']),
            'latexmlc': _run_command(['latexmlc', '--version']),
        },
        'texlive': {
            'cjk_font_family': cjk_font_family,
            'cjk_sty': _read_tex_file_metadata(cjk_sty_path),
            'cjk_font_fd': _read_tex_file_metadata(cjk_fd_path),
        },
    }


def build_eval_run_report(save_name, result_all, page_denominators):
    return {
        'save_name': save_name,
        'runtime_environment': collect_runtime_environment_report(),
        'stage_execution': build_stage_execution_summary(result_all),
        'page_denominators': page_denominators,
        'notebook_metric_summary': build_notebook_metric_summary(result_all, page_denominators),
    }


def format_eval_run_report(report):
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)


def format_runtime_environment_log(runtime_environment, save_name=''):
    lines = []
    if save_name:
        lines.append(f'save_name: {save_name}')

    system = (runtime_environment or {}).get('system', {})
    if system:
        lines.append('[system]')
        for key in ['platform', 'python_version', 'python_executable', 'python_prefix', 'conda_env', 'conda_prefix', 'libc', 'uname']:
            lines.append(f'{key}: {system.get(key, "")}')
        os_release = system.get('os_release', {})
        if os_release:
            lines.append(f'os_release: {json.dumps(os_release, ensure_ascii=False, sort_keys=True)}')

    system_packages = (runtime_environment or {}).get('system_packages', {})
    if system_packages:
        lines.append('[system_packages]')
        lines.append(f'package_manager: {system_packages.get("package_manager")}')
        for package_name, version in sorted((system_packages.get('tracked_packages') or {}).items()):
            lines.append(f'{package_name}: {version}')

    python_packages = (runtime_environment or {}).get('python_packages', {})
    if python_packages:
        lines.append('[python_packages]')
        for package_name, version in sorted(python_packages.items()):
            lines.append(f'{package_name}: {version}')

    external_tools = (runtime_environment or {}).get('external_tools', {})
    if external_tools:
        lines.append('[external_tools]')
        for tool_name, payload in sorted(external_tools.items()):
            if isinstance(payload, dict):
                lines.append(
                    f'{tool_name}: path={payload.get("path")} returncode={payload.get("returncode")} output={payload.get("output")}'
                )
            else:
                lines.append(f'{tool_name}: {payload}')

    texlive = (runtime_environment or {}).get('texlive', {})
    if texlive:
        lines.append('[texlive]')
        lines.append(f'cjk_font_family: {texlive.get("cjk_font_family")}')
        for key in ['cjk_sty', 'cjk_font_fd']:
            payload = texlive.get(key, {})
            if isinstance(payload, dict):
                lines.append(
                    f'{key}: status={payload.get("status")} path={payload.get("path")} version={payload.get("version")} date={payload.get("date")} header={payload.get("header")}'
                )
                if payload.get('error'):
                    lines.append(f'{key}_error: {payload.get("error")}')
            else:
                lines.append(f'{key}: {payload}')

    if not lines:
        lines.append('No runtime environment info recorded.')
    return '\n'.join(lines) + '\n'


def format_stage_execution_log(stage_execution, save_name=''):
    lines = []
    if save_name:
        lines.append(f'save_name: {save_name}')

    page_match = (stage_execution or {}).get('page_match', {})
    if page_match:
        lines.append('[page_match]')
        lines.append(f'workers: {page_match.get("workers")}')
        lines.append(f'page_count: {page_match.get("page_count")}')
        lines.append(f'quick_match_truncated_timeout_sec: {page_match.get("quick_match_truncated_timeout_sec")}')
        lines.append(f'match_timeout_sec: {page_match.get("match_timeout_sec")}')
        fallbacks = page_match.get('fallbacks', {})
        if fallbacks:
            lines.append('fallbacks:')
            for reason, payload in sorted(fallbacks.items()):
                cases = payload.get('cases', [])
                lines.append(f'  - {reason}: count={payload.get("count", len(cases))}')
                for case_name in cases:
                    lines.append(f'    case: {case_name}')

    metric_groups = (stage_execution or {}).get('metrics', {})
    for element in sorted(metric_groups.keys()):
        metrics = metric_groups.get(element, {})
        for metric_name in sorted(metrics.keys()):
            payload = metrics[metric_name]
            lines.append(f'[{element}.{metric_name}]')
            lines.append(f'workers: {payload.get("workers")}')
            if "timeout_sec" in payload:
                lines.append(f'timeout_sec: {payload.get("timeout_sec")}')
            lines.append(f'sample_count: {payload.get("sample_count")}')
            lines.append(f'timeout_case_count: {payload.get("timeout_case_count")}')
            for case in payload.get('timeout_cases', []):
                lines.append(f'  timeout_case: {case.get("case_name")} reason={case.get("reason", "")}'.rstrip())
            lines.append(f'error_case_count: {payload.get("error_case_count")}')
            for case in payload.get('error_cases', []):
                lines.append(f'  error_case: {case.get("case_name")} reason={case.get("reason", "")}'.rstrip())
            lines.append(f'exception_case_count: {payload.get("exception_case_count")}')
            for case in payload.get('exception_cases', []):
                lines.append(f'  exception_case: {case.get("case_name")} reason={case.get("reason", "")}'.rstrip())

    if not lines:
        lines.append('No stage execution debug info recorded.')
    return '\n'.join(lines) + '\n'
