import os
import shutil
import subprocess


PREFERRED_TEXLIVE_BIN_DIRS = [
    "/share/texlive",
    "/usr/local/texlive/2025/bin/x86_64-linux",
]
DEFAULT_CJK_FONT_FAMILY = "gkai"
_TEXLIVE_WEB2C_SUFFIX = os.path.join("texmf-dist", "web2c")


def _read_first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _which_tex_binary(binary_name: str) -> str:
    return shutil.which(binary_name) or ""


def _is_executable_file(path: str) -> bool:
    return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def _is_texlive_root(path: str) -> bool:
    normalized = (path or "").strip()
    if not normalized or not os.path.isdir(normalized):
        return False
    return os.path.isdir(os.path.join(normalized, "texmf-dist")) or os.path.isdir(os.path.join(normalized, "tlpkg"))


def _dedupe_paths(paths: list[str]) -> list[str]:
    unique_paths = []
    for path in paths:
        normalized = (path or "").strip()
        if normalized and normalized not in unique_paths:
            unique_paths.append(normalized)
    return unique_paths


def _iter_bin_subdirs(root: str) -> list[str]:
    bin_root = os.path.join(root, "bin")
    if not os.path.isdir(bin_root):
        return []

    subdirs = []
    try:
        for entry in sorted(os.listdir(bin_root)):
            candidate = os.path.join(bin_root, entry)
            if os.path.isdir(candidate):
                subdirs.append(candidate)
    except OSError:
        return []
    return subdirs


def _iter_texlive_year_roots(prefix: str) -> list[str]:
    normalized = (prefix or "").strip()
    if not normalized or not os.path.isdir(normalized):
        return []

    year_roots = []
    try:
        for entry in sorted(os.listdir(normalized), reverse=True):
            if not entry.isdigit():
                continue
            candidate = os.path.join(normalized, entry)
            if _is_texlive_root(candidate):
                year_roots.append(candidate)
    except OSError:
        return []
    return year_roots


def _guess_texlive_root_from_bin_dir(bin_dir: str) -> str:
    normalized = (bin_dir or "").strip()
    if not normalized:
        return ""

    candidate = os.path.dirname(os.path.dirname(normalized))
    if _is_texlive_root(candidate):
        return candidate
    return ""


def _discover_texlive_layouts(prefix: str) -> list[dict[str, str]]:
    normalized = (prefix or "").strip()
    if not normalized:
        return []

    layouts = []
    seen = set()

    def _append_layout(root: str, bin_dir: str) -> None:
        normalized_root = (root or "").strip()
        normalized_bin = (bin_dir or "").strip()
        if not normalized_bin:
            return
        key = (normalized_root, normalized_bin)
        if key in seen:
            return
        seen.add(key)
        layouts.append({
            "root": normalized_root,
            "bin_dir": normalized_bin,
        })

    if _is_texlive_root(normalized):
        for candidate_bin_dir in _iter_bin_subdirs(normalized):
            _append_layout(normalized, candidate_bin_dir)

    for year_root in _iter_texlive_year_roots(normalized):
        for candidate_bin_dir in _iter_bin_subdirs(year_root):
            _append_layout(year_root, candidate_bin_dir)

    guessed_root = _guess_texlive_root_from_bin_dir(normalized)
    if guessed_root:
        _append_layout(guessed_root, normalized)

    if _is_executable_file(os.path.join(normalized, "pdflatex")) or _is_executable_file(os.path.join(normalized, "kpsewhich")):
        wrapper_root = ""
        year_roots = _iter_texlive_year_roots(normalized)
        if year_roots:
            wrapper_root = year_roots[0]
        _append_layout(wrapper_root, normalized)

    return layouts


def _candidate_texlive_prefixes() -> list[str]:
    prefixes: list[str] = []

    explicit_root = _read_first_env("CDM_TEXLIVE_ROOT", "OMNIDOCBENCH_TEXLIVE_ROOT")
    if explicit_root:
        prefixes.append(explicit_root)

    explicit_pdflatex = _read_first_env("CDM_PDFLATEX", "OMNIDOCBENCH_PDFLATEX")
    if explicit_pdflatex:
        prefixes.append(os.path.dirname(explicit_pdflatex))

    explicit_kpsewhich = _read_first_env("CDM_KPSEWHICH", "OMNIDOCBENCH_KPSEWHICH")
    if explicit_kpsewhich:
        prefixes.append(os.path.dirname(explicit_kpsewhich))

    explicit_bin = _read_first_env("CDM_TEXLIVE_BIN", "OMNIDOCBENCH_TEXLIVE_BIN")
    if explicit_bin:
        prefixes.append(explicit_bin)

    prefixes.extend(PREFERRED_TEXLIVE_BIN_DIRS)
    prefixes = _dedupe_paths(prefixes)

    has_preferred_layout = any(_discover_texlive_layouts(prefix) for prefix in prefixes)
    if has_preferred_layout:
        return prefixes

    which_pdflatex = _which_tex_binary("pdflatex")
    if which_pdflatex:
        prefixes.append(os.path.dirname(which_pdflatex))

    which_kpsewhich = _which_tex_binary("kpsewhich")
    if which_kpsewhich:
        prefixes.append(os.path.dirname(which_kpsewhich))

    return _dedupe_paths(prefixes)


def _has_explicit_texlive_selection() -> bool:
    return any(
        _read_first_env(env_name)
        for env_name in [
            "CDM_TEXLIVE_ROOT",
            "OMNIDOCBENCH_TEXLIVE_ROOT",
            "CDM_TEXLIVE_BIN",
            "OMNIDOCBENCH_TEXLIVE_BIN",
            "CDM_PDFLATEX",
            "OMNIDOCBENCH_PDFLATEX",
            "CDM_KPSEWHICH",
            "OMNIDOCBENCH_KPSEWHICH",
        ]
    )


def _should_avoid_system_tex_fallback(texlive_bin_dir: str) -> bool:
    normalized = (texlive_bin_dir or "").strip()
    return _has_explicit_texlive_selection() or normalized.startswith("/share/texlive")


def _resolve_texlive_layout(required_binary: str = "") -> dict[str, str]:
    layouts = []
    seen = set()
    for prefix in _candidate_texlive_prefixes():
        for layout in _discover_texlive_layouts(prefix):
            key = (layout.get("root", ""), layout.get("bin_dir", ""))
            if key in seen:
                continue
            seen.add(key)
            layouts.append(layout)

    def _layout_has_binary(layout: dict[str, str], binary_name: str) -> bool:
        return _is_executable_file(os.path.join(layout.get("bin_dir", ""), binary_name))

    for layout in layouts:
        if _layout_has_binary(layout, "pdflatex") and _layout_has_binary(layout, "kpsewhich"):
            return layout

    if required_binary:
        for layout in layouts:
            if _layout_has_binary(layout, required_binary):
                return layout

    return {}


def _candidate_texlive_bin_dirs() -> list[str]:
    candidates: list[str] = []
    for layout in (_discover_texlive_layouts(prefix) for prefix in _candidate_texlive_prefixes()):
        for item in layout:
            candidates.append(item.get("bin_dir", ""))
    return _dedupe_paths(candidates)


def _resolve_texlive_bin_dir() -> str:
    explicit_pdflatex = _read_first_env("CDM_PDFLATEX", "OMNIDOCBENCH_PDFLATEX")
    if explicit_pdflatex:
        layout = _resolve_texlive_layout("pdflatex")
        if layout:
            return layout.get("bin_dir", os.path.dirname(explicit_pdflatex))
        return os.path.dirname(explicit_pdflatex)

    explicit_bin = _read_first_env("CDM_TEXLIVE_BIN", "OMNIDOCBENCH_TEXLIVE_BIN")
    if explicit_bin:
        layout = _resolve_texlive_layout("pdflatex")
        if layout:
            return layout.get("bin_dir", explicit_bin)
        return explicit_bin

    layout = _resolve_texlive_layout("pdflatex")
    if layout:
        return layout.get("bin_dir", "")

    which_pdflatex = _which_tex_binary("pdflatex")
    if which_pdflatex:
        return os.path.dirname(which_pdflatex)

    return PREFERRED_TEXLIVE_BIN_DIRS[0]


def resolve_tex_binary(binary_name: str) -> str:
    override = _read_first_env(
        f"CDM_{binary_name.upper()}",
        f"OMNIDOCBENCH_{binary_name.upper()}",
    )
    if override:
        return override

    layout = _resolve_texlive_layout(binary_name)
    texlive_bin_dir = layout.get("bin_dir", "") if layout else _resolve_texlive_bin_dir()
    candidate = os.path.join(texlive_bin_dir, binary_name) if texlive_bin_dir else ""
    if _is_executable_file(candidate):
        return candidate
    if candidate and _should_avoid_system_tex_fallback(texlive_bin_dir):
        return candidate

    which_binary = _which_tex_binary(binary_name)
    if which_binary:
        return which_binary

    return candidate


def build_tex_env() -> dict:
    env = os.environ.copy()

    path_items = []
    layout = _resolve_texlive_layout()
    texlive_bin_dir = layout.get("bin_dir", "") if layout else _resolve_texlive_bin_dir()
    if texlive_bin_dir:
        path_items.append(texlive_bin_dir)

    existing_path = env.get("PATH", "")
    if existing_path:
        for item in existing_path.split(os.pathsep):
            if item and item not in path_items:
                path_items.append(item)

    if path_items:
        env["PATH"] = os.pathsep.join(path_items)

    texlive_root = layout.get("root", "") if layout else _guess_texlive_root_from_bin_dir(texlive_bin_dir)
    texmfcnf = env.get("TEXMFCNF", "").strip()
    if texlive_root and not texmfcnf:
        web2c_dir = os.path.join(texlive_root, _TEXLIVE_WEB2C_SUFFIX)
        if os.path.isdir(web2c_dir):
            env["TEXMFCNF"] = os.pathsep.join([texlive_root, web2c_dir])

    if texlive_bin_dir:
        env.setdefault("CDM_TEXLIVE_BIN", texlive_bin_dir)
    if texlive_root:
        env.setdefault("CDM_TEXLIVE_ROOT", texlive_root)
    resolved_pdflatex = resolve_tex_binary("pdflatex")
    if resolved_pdflatex:
        env.setdefault("CDM_PDFLATEX", resolved_pdflatex)
    resolved_kpsewhich = resolve_tex_binary("kpsewhich")
    if resolved_kpsewhich:
        env.setdefault("CDM_KPSEWHICH", resolved_kpsewhich)

    return env


def resolve_cjk_font_family() -> str:
    return os.getenv("CDM_CJK_FONT", "").strip() or DEFAULT_CJK_FONT_FAMILY


def _run_text_command(argv: list[str], *, timeout: int = 5) -> str:
    try:
        completed = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
            env=build_tex_env(),
        )
        lines = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
        if lines:
            return lines[0]
        return f"exit_code={completed.returncode}"
    except Exception as exc:
        return f"unavailable ({type(exc).__name__}: {exc})"


def _kpsewhich_lookup(target: str) -> str:
    kpsewhich_path = resolve_tex_binary("kpsewhich")
    return _run_text_command([kpsewhich_path, target])


def describe_tex_runtime() -> dict[str, str]:
    layout = _resolve_texlive_layout()
    texlive_bin_dir = layout.get("bin_dir", "") if layout else _resolve_texlive_bin_dir()
    texlive_root = layout.get("root", "") if layout else _guess_texlive_root_from_bin_dir(texlive_bin_dir)
    pdflatex_path = resolve_tex_binary("pdflatex")
    kpsewhich_path = resolve_tex_binary("kpsewhich")
    tex_env = build_tex_env()

    return {
        "texlive_root": texlive_root,
        "texlive_bin_dir": texlive_bin_dir,
        "pdflatex": pdflatex_path,
        "pdflatex_version": _run_text_command([pdflatex_path, "--version"]),
        "kpsewhich": kpsewhich_path,
        "kpsewhich_texmfroot": _run_text_command([kpsewhich_path, "-var-value=TEXMFROOT"]),
        "texmfcnf": tex_env.get("TEXMFCNF", ""),
        "cjk_font": resolve_cjk_font_family(),
        "cjk_sty": _kpsewhich_lookup("CJK.sty"),
        "cjk_font_fd": _kpsewhich_lookup(f"c70{resolve_cjk_font_family()}.fd"),
    }
