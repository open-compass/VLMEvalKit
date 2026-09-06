import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

_MOLECULE_BLOCKLIST = frozenset({
    'molecule', 'smiles', 'selfies', 'iupac', 'chemical', 'answer', 'formula', 'structure', 'compound',
    'representation', 'format', 'output', 'result', 'here', 'is', 'the', 'below', 'above', 'following', 'see'
})
_SELFIES_FULL_RE = re.compile('(?:\\[[A-Za-z0-9_@+\\-=#$\\\\/%.:*]+\\])+')
_ELEMENT_SYMBOLS = 'Ac|Ag|Al|Am|Ar|As|At|Au|Ba|Be|Bh|Bi|Bk|Br|Ca|Cd|Ce|Cf|Cl|Cm|Cn|Co|Cr|Cs|Cu|Db|Ds|Dy|Er|Es|Eu|Fe|Fl|Fm|Fr|Ga|Gd|Ge|He|Hf|Hg|Ho|Hs|In|Ir|Kr|La|Li|Lr|Lu|Lv|Mc|Md|Mg|Mn|Mo|Mt|Na|Nb|Nd|Ne|Nh|Ni|No|Np|Os|Pa|Pb|Pd|Pm|Po|Pr|Pt|Pu|Ra|Rb|Re|Rf|Rg|Rh|Rn|Ru|Sb|Sc|Se|Sg|Si|Sm|Sn|Sr|Ta|Tb|Tc|Te|Th|Ti|Tl|Tm|Ts|Xe|Yb|Zn|Zr|B|C|N|O|F|P|S|I|K|V|Y|W|H|U|b|c|n|o|p|s'  # noqa: E501
_SMILES_TOKEN_RE = re.compile(  # noqa: E501
    f'\\[[^\\[\\]\\s]+\\]|{_ELEMENT_SYMBOLS}|\\d+|[@+\\-\\(\\)=#$\\\\/%.:*]')
_FINAL_ANSWER_MARKER_RE = re.compile(
    'final[\\s_-]+(?:numerical[\\s_-]+)?answer\\s*[:：]?|the\\s+answer\\s+is\\s*(?:[:：]|\\.(?=\\s|$))?|(?<!final\\s)\\banswer\\s+is\\s*(?:[:：]|\\.(?=\\s|$))?|final\\s+conclusion\\s*[:：]?|\\bresult\\s+is\\s*(?:[:：]|\\.(?=\\s|$))?',  # noqa: E501
    re.I)


def _looks_like_selfies(s: str) -> bool:
    if not _SELFIES_FULL_RE.fullmatch(s):
        return False
    return bool(re.search('\\[(?:[=#]?)(?:Cl|Br|Si|Se|Te|[BCNOFPSIbcnosp])', s))


def _looks_like_smiles_without_rdkit(s: str) -> bool:
    tokens = _SMILES_TOKEN_RE.findall(s)
    if not tokens or ''.join(tokens) != s:
        return False
    return any((t.startswith('[') or re.fullmatch(_ELEMENT_SYMBOLS, t) for t in tokens))


def _is_valid_molecule_candidate(s: str) -> bool:
    if not s:
        return False
    s = s.strip().strip('"\'`')
    low = s.lower()
    if low in _MOLECULE_BLOCKLIST:
        return False
    if _looks_like_selfies(s):
        return True
    if not re.fullmatch('[A-Za-z0-9@+\\-\\[\\]\\(\\)=#$\\\\/%.:*]+', s):
        return False
    if not re.search('(?:Cl|Br|Si|Se|Te|Na|Li|Mg|Ca|Al|[BCNOFPSIbcnosp])', s):
        return False
    try:
        from .rdkit_utils import mol_from_smiles_lenient
        return mol_from_smiles_lenient(s) is not None
    except Exception:
        return _looks_like_smiles_without_rdkit(s)


def _walk_json_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, dict):
        for key in ('answer', 'smiles', 'selfies', 'molecule', 'result'):
            if key in obj:
                yield from _walk_json_strings(obj[key])
        for key, value in obj.items():
            if str(key).lower() not in {'answer', 'smiles', 'selfies', 'molecule', 'result'}:
                yield from _walk_json_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_json_strings(value)
    elif isinstance(obj, str):
        yield obj


def _json_string_candidates(text: str) -> List[str]:
    out: List[str] = []
    snippets = [(text or '').strip()]
    snippets.extend((m.group(1).strip() for m in re.finditer('```(?:json)?\\s*([\\s\\S]*?)```', text or '', re.I)))
    for snippet in snippets:
        try:
            obj = json.loads(snippet)
        except Exception:
            continue
        out.extend(_walk_json_strings(obj))
    return out


def _segment_molecule_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    candidates.extend(_json_string_candidates(text))
    stripped = text.strip().strip('"\'`').rstrip('.,;:')
    if stripped and (not re.search('\\s', stripped)):
        candidates.append(stripped)
    fence_matches = list(re.finditer('```(?:smiles|selfies|text)?\\s*([\\s\\S]*?)```', text, re.I))
    for m in reversed(fence_matches):
        block = m.group(1).strip()
        if block:
            candidates.append(block)
    label_re = re.compile(
        '(?:canonical\\s+)?(?:SMILES|SELFIES)\\s*[:：=]\\s*(?:`([^`]+)`|\\"([^\\"]+)\\"|\'([^\']+)\'|([^\\s,;]+))', re.I)
    for m in reversed(list(label_re.finditer(text))):
        value = next((g for g in m.groups() if g), '').strip()
        if value:
            candidates.append(value)
    selfies_matches = _SELFIES_FULL_RE.findall(text)
    candidates.extend(sorted(selfies_matches, key=len, reverse=True))
    token_re = re.compile('[A-Za-z0-9@+\\-\\[\\]\\(\\)=#$\\\\/%.:*]{3,}')
    candidates.extend(sorted(token_re.findall(text), key=len, reverse=True))
    return candidates


def _molecule_candidates(response: str) -> List[str]:
    text = response or ''
    segments: List[str] = []
    markers = list(_FINAL_ANSWER_MARKER_RE.finditer(text))
    if markers:
        tail = text[markers[-1].end():].strip()
        if tail:
            segments.append(tail)
    segments.append(text)
    candidates: List[str] = []
    for segment in segments:
        candidates.extend(_segment_molecule_candidates(segment))
    out: List[str] = []
    seen = set()
    for raw in candidates:
        value = raw.strip().strip('"\'`').rstrip('.,;:')
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def extract_molecule(response: str, llm_client=None, item: Optional[Dict] = None) -> Tuple[str, str, str]:
    for candidate in _molecule_candidates(response):
        if not _is_valid_molecule_candidate(candidate):
            continue
        fmt = 'selfies' if _looks_like_selfies(candidate) else 'smiles'
        return (candidate, fmt, 'structured/regex')
    if llm_client:
        prompt_addendum = ('Extract the chemical molecule representation (SMILES or SELFIES format) from the response. '
                           'Return only the molecule string, nothing else.')
        extracted = _extract_via_llm(
            response,
            llm_client,
            prompt_addendum=prompt_addendum,
        )
        if extracted and _is_valid_molecule_candidate(extracted):
            fmt = 'selfies' if extracted.strip().startswith('[') else 'smiles'
            return (extracted.strip(), fmt, 'llm')
    return ('', '', '')


def extract_method(response: str,
                   required_parameters: str,
                   llm_client=None,
                   item: Optional[Dict] = None) -> Tuple[str, str]:
    if llm_client:
        extracted = _extract_via_llm(
            response,
            llm_client,
            prompt_addendum=(
                f'The required formula or law is: {required_parameters}. '
                'Extract only the formula, law, or scientific method that the model response actually uses. '
                'Do not infer or supply a method that is not explicitly present in the response. '
                'If no method is present, return NOT_FOUND. Otherwise, return only the extracted '
                'formula/method string, with no explanation.'),
        )
        if extracted and extracted.strip().upper() != 'NOT_FOUND':
            return (extracted.strip(), 'llm')
    method_str = _extract_method_regex(response, required_parameters)
    if method_str:
        return (method_str, 'regex')
    return ('', '')


def _extract_molecule_regex(response: str) -> Tuple[str, str]:
    selfies_matches = re.findall('(?:\\[[^\\]]+\\])+', response)
    if selfies_matches:
        full = max(selfies_matches, key=len)
        if full and '[' in full and (']' in full) and _is_valid_molecule_candidate(full):
            return (full, 'selfies')
    for m in re.finditer('[C][^\\s\\[\\]]{5,}', response):
        cand = m.group(0)
        if re.match('^[CBNOSPFI\\[\\]ClBr@=\\-\\(\\)\\\\\\/\\.#+\\d]+$', cand):
            return (cand, 'smiles')
    if selfies_matches:
        full = max(selfies_matches, key=len)
        if full and '[' in full and (']' in full):
            return (full, 'selfies')
    return ('', '')


def _extract_method_regex(response: str, required_parameters: str) -> str:
    text = required_parameters.strip()
    for phrase in [
            'specific chemical fomulas or laws', 'specific physical fomulas or laws',
            'specific geographical fomulas or laws', 'specific biological fomulas or laws',
            'specific materials science fomulas or laws'
    ]:
        text = re.sub(re.escape(phrase) + '\\s*[:：]?\\s*', '', text, flags=re.I)
    text = text.strip()
    if not text:
        return ''
    if re.search(re.escape(text), response, re.I):
        return text
    if '=' in text:
        parts = text.split('=', 1)
        if len(parts) == 2:
            pat = re.escape(parts[0].strip()) + '\\s*=\\s*' + re.escape(parts[1].strip())
            m = re.search(pat, response, re.I)
            if m:
                return m.group(0)
    return ''


def _extract_via_llm(response: str, llm_client, prompt_addendum: str) -> str:
    prompt = ('Extract the requested content from the following model response.\n\n'
              f'{prompt_addendum}\n\nModel response:\n{response[:2500]}\n\n'
              'Extracted content (only the extracted string, no explanation):')
    try:
        if hasattr(llm_client, 'chat'):
            result = llm_client.chat(prompt)
        elif hasattr(llm_client, 'complete'):
            result = llm_client.complete(prompt)
        else:
            result = llm_client(prompt)
        if not isinstance(result, str):
            result = getattr(result, 'content', None) or getattr(result, 'text', None) or str(result)
        result = result or ''
        lines = result.strip().split('\n')
        for line in lines:
            line = line.strip().strip('"\'`')
            if line and len(line) > 2:
                return line
        return result.strip().strip('"\'`') if result else ''
    except Exception:
        return ''
