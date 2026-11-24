"""Rule-based scorers for ReasonMap and ReasonMap-Plus.

Ported from the official evaluation code (https://github.com/fscdc/ReasonMap, ``reasonmap/``),
so scores computed here match ``python score.py`` in that repository.

ReasonMap (route planning, short and long questions)
    * ``acc``: the answer starts at station_1, ends at station_2, every section's line exists in the
      station table and contains both of its stops, and consecutive sections share their transfer stop.
    * ``map score`` (short, max 10 + 10): against each reference route plan, +1 for correct endpoints,
      then per section +2 line name, +1 departure stop, +1 arrival stop; capped at 10, +10 when ``acc``.
    * ``map score`` (long, max 20 + 20): as above plus a via-stop term ("Number of Via Stops": up to 4
      per section by relative error; "Via Stops": (IoU * 10 + hits) / 2); capped at 20, +20 when ``acc``.
    * Difficulty-aware weighting (paper, Appendix B.3): weight = 1 + 0.5 * (q + m), where q / m are the
      question / map difficulty levels (easy 0, middle 1, hard 2).

ReasonMap-Plus (counting and true/false questions)
    * The short answer is pulled from ``\\boxed{}`` or from "the answer is ..." and compared exactly.
    * Weighted accuracy uses the map difficulty (easy 1.0, middle 1.5, hard 2.0).
"""

import re

# ----------------------------------------------------------------------------- common

THINK_END_MARKERS = ('</think>', '◁/think▷')

DIFFICULTY_LEVELS = {'easy': 0, 'middle': 1, 'medium': 1, 'hard': 2}
PLUS_DIFFICULTY_WEIGHTS = {'easy': 1.0, 'middle': 1.5, 'hard': 2.0}

STATION_MARKERS = ('（换乘站）', '（支线起始站）', ' (Transfer Station)', ' (Branch-starting Station)')


def strip_reasoning(text):
    """Drop a reasoning trace (``<think>...</think>``, Kimi-VL ``◁think▷...◁/think▷``) and return the answer."""
    text = '' if text is None else str(text)
    for marker in THINK_END_MARKERS:
        if marker in text:
            text = text.split(marker)[-1]
    return text.strip()


def reasonmap_weight(difficulty_question, difficulty_city):
    """Weight of a (question difficulty, map difficulty) pair, see the paper's Appendix B.3."""
    q = DIFFICULTY_LEVELS.get(str(difficulty_question).lower(), 0)
    m = DIFFICULTY_LEVELS.get(str(difficulty_city).lower(), 0)
    return 1.0 + 0.5 * (q + m)


def plus_weight(difficulty_city):
    return PLUS_DIFFICULTY_WEIGHTS.get(str(difficulty_city).lower(), 1.0)


def clean_station_table(metro_data):
    """Route name -> station list, with the transfer / branch markers removed."""
    cleaned = {}
    for route_name, stations in metro_data.items():
        names = []
        for s in stations:
            for marker in STATION_MARKERS:
                s = s.replace(marker, '')
            names.append(s)
        cleaned[route_name] = names
    return cleaned


def clean_string(s):
    """Normalise a station / route string: drop "Metro", commas, ``*``, newlines and parenthesised
    parts, then remove whitespace / dashes and lower-case."""
    if s is None:
        return ''
    s = str(s)
    s = s.replace('Metro', '').replace('metro', '').replace(',', '').replace('*', '').replace('\n', '')
    s = re.sub(r'[\(\（][^\)\）]*[\)\）]', '', s)
    return re.sub(r'[\s\-]+', '', s).lower()


# ----------------------------------------------------------------------------- ReasonMap: parsing

SECTION_SEPARATOR = '--'

# Reference line-name aliases that can't be derived by pattern (city -> {api name: station key}).
_ROUTE_ALIASES = {
    'auckland': {'West': 'Western Line', 'Sth': 'Southern Line', 'East': 'Eastern Line'},
    'dubai': {'MRed1': 'Red Line', 'MRed2': 'Red Line', 'MGrn': 'Green Line'},
    'kl': {
        'MRT Putrajaya Line': 'Line 12',
        'Ampang Line': 'Line 3',
        'LRT Kelana Jaya Line': 'Line 5',
        'KTM Tanjung Malim - Pelabuhan Klang': 'Line 2',
        'LRT Sri Petaling Line': 'Line 4',
        'MRT Kajang Line': 'Line 9',
        'KTM Batu Caves - Pulau Sebang/Tampin': 'Line 1',
    },
    'rome': {'B1': 'Line B'},  # B1 is a branch of line B
}

# Patterns that extract a line identifier from a reference (map API) route name.
_ROUTE_PATTERNS = [
    re.compile(r'^Metro\s+([A-Z0-9]{1,2})[-\s]?Line$', re.IGNORECASE),  # LA: "Metro A Line", "Metro B-Line"
    re.compile(r'^(?:Metro|Linea)\s+([A-Z0-9]{1,2})$', re.IGNORECASE),  # Rome: "Metro B", "Linea C"
    re.compile(r'^M(\d+)\b.*metróvonal$', re.IGNORECASE),  # Budapest: "M1-es Metróvonal"
    re.compile(r'^([A-Z0-9]{1,2})\s+(?:Line|Train)$', re.IGNORECASE),  # NYC: "1 Line", "Q Train"
    re.compile(r'^([A-Z0-9]{1,2})$'),  # Oslo / Toronto / Geneva "1" / "18", Rome "A", Sydney "M1"
]


def _field(section, label, exclude_prefix=None):
    """Value of ``label:`` on its own line inside ``section`` (None when absent)."""
    lookbehind = f'(?<!{re.escape(exclude_prefix)})' if exclude_prefix else ''
    match = re.search(lookbehind + re.escape(label) + r':[ \t]*([^\n]*)', section)
    if not match:
        return None
    return match.group(1).strip()


def _split_via_stops(value):
    if not value:
        return []
    items = [v.strip() for v in re.split(r'[,，、;；]', value)]
    return [v for v in items if v and v.lower() not in ('none', 'n/a', '无')]


def parse_route_sections(answer, via_mode=None):
    """Parse an answer into route sections.

    ``via_mode`` is None for short questions, "num" for "Number of Via Stops" and "list" for
    "Via Stops". Sections missing one of the mandatory labels are skipped.
    """
    sections = []
    for section in (answer or '').split(SECTION_SEPARATOR):
        if not section.strip():
            continue
        if 'Route Name:' not in section or 'Departure Stop:' not in section or 'Arrival Stop:' not in section:
            continue
        route_name = _field(section, 'Route Name')
        departure = _field(section, 'Departure Stop')
        arrival = _field(section, 'Arrival Stop')
        info = {
            'Route Name': route_name if route_name else 'Wrong',
            'Departure Stop': clean_string(departure) if departure else 'Wrong',
            'Arrival Stop': clean_string(arrival) if arrival else 'Wrong',
        }
        if via_mode == 'num':
            value = _field(section, 'Number of Via Stops') or ''
            match = re.search(r'\d+', value)
            info['Number of Via Stops'] = int(match.group()) if match else 0
        elif via_mode == 'list':
            value = _field(section, 'Via Stops', exclude_prefix='Number of ')
            info['Via Stops'] = [clean_string(v) for v in _split_via_stops(value)]
        sections.append(info)
    return sections


def via_mode_for_question(question_long):
    return 'num' if 'Number of Via Stops:' in question_long else 'list'


def unbox_route_answer(text):
    """QVQ-style answers put the whole route inside ``\\boxed{\\text{...}}``; return it as plain text.

    Only used when the plain answer has no parsable route section.
    """
    match = re.search(r'boxed{(.+)}', text, flags=re.S)
    if not match:
        return text
    boxed = match.group(1)
    parts = re.findall(r'\\text{(.+?)}', boxed)
    return '\n'.join(parts) if parts else boxed


def extract_route_answer(prediction):
    answer = strip_reasoning(prediction)
    if not parse_route_sections(answer) and 'boxed{' in answer:
        answer = unbox_route_answer(answer)
    return answer


# ----------------------------------------------------------------------------- ReasonMap: normalisation


def normalize_answer_route_name(name):
    """Beijing sub-line aliases accepted in answers (八通线 -> Line 1, 大兴线 -> Line 4)."""
    if '八通' in name:
        return 'Line 1'
    if '大兴' in name:
        return 'Line 4'
    return name


def normalize_reference_route_name(name, country, city, station_keys=()):
    """Map a reference (map API) route name onto the naming used by the station tables."""
    name = (name or '').strip()
    if country == 'china':
        # 地铁5号线(后关--虎滩新区) -> Line 5 ; 地铁昌平线(昌平西山口--蓟门桥) -> 昌平线
        base = re.sub(r'\(.*?\)|（.*?）', '', name)
        match = re.search(r'(\d+)', base)
        if match:
            return f'Line {int(match.group(1))}'
        return base.replace('地铁', '').strip()

    keys = set(station_keys)
    if name in keys:
        return name
    alias = _ROUTE_ALIASES.get(city, {}).get(name)
    if alias:
        return alias
    for pattern in _ROUTE_PATTERNS:
        match = pattern.match(name)
        if match:
            ident = match.group(1).upper()
            for candidate in (f'Line {ident}', ident):
                if candidate in keys:
                    return candidate
            return f'Line {ident}'
    return name  # e.g. regional trains that are not on the metro map


def reference_stop_name(name, country):
    """Reference stops come without suffix; answers use "xx站" / "xx Station"."""
    name = (name or '').strip()
    suffix = '站' if country == 'china' else 'Station'
    if name.endswith(suffix):
        return name
    return name + suffix if country == 'china' else f'{name} {suffix}'


# ----------------------------------------------------------------------------- ReasonMap: metrics


def _endpoints_match(route_data, station_1, station_2):
    return (clean_string(route_data[0]['Departure Stop']) == clean_string(station_1)
            and clean_string(route_data[-1]['Arrival Stop']) == clean_string(station_2))


def route_is_correct(route_data, station_1, station_2, metro_data):
    """``acc``: 1 if the answered route is a valid route between the two stations."""
    if not route_data or not _endpoints_match(route_data, station_1, station_2):
        return 0
    for i, section in enumerate(route_data):
        route_name = normalize_answer_route_name(section['Route Name'])
        if route_name not in metro_data:
            return 0
        stations = [clean_string(s) for s in metro_data[route_name]]
        departure, arrival = section['Departure Stop'], section['Arrival Stop']
        if clean_string(departure) not in stations or clean_string(arrival) not in stations:
            return 0
        if i < len(route_data) - 1 and clean_string(arrival) != clean_string(route_data[i + 1]['Departure Stop']):
            return 0
    return 1


def _reference_plans(routes):
    if not routes:
        return []
    return [plan for plan in routes.values() if plan]


def _compare_section(section, ref, country, city, station_keys):
    """Line-name / stop points for one answered section against one reference section."""
    ref_name = normalize_reference_route_name(ref['route_name'], country, city, station_keys)
    ref_departure = clean_string(reference_stop_name(ref['departure_stop'], country))
    ref_arrival = clean_string(reference_stop_name(ref['arrival_stop'], country))
    name_ok = clean_string(ref_name) == clean_string(section['Route Name'])
    dep_ok = ref_departure == clean_string(section['Departure Stop'])
    arr_ok = ref_arrival == clean_string(section['Arrival Stop'])
    return ref_name, name_ok, dep_ok, arr_ok


def score_short(answer, meta, metro_data):
    """Score a short-question answer. ``meta`` needs country, city, station_1, station_2 and routes."""
    country, city = meta['country'], meta['city']
    station_keys = list(metro_data)
    result = {'acc': 0, 'map_score': 0, 'departure_arrival': 0, 'route_name': 0, 'stops': 0}

    route_data = parse_route_sections(answer)
    if not route_data:
        return result

    acc = route_is_correct(route_data, meta['station_1'], meta['station_2'], metro_data)
    result['acc'] = acc
    endpoints_ok = _endpoints_match(route_data, meta['station_1'], meta['station_2'])

    best = None
    for plan in _reference_plans(meta.get('routes')):
        score = dep_arr = name_pts = stop_pts = 0
        if endpoints_ok:
            score += 1
            dep_arr += 1
            for section, ref in zip(route_data, plan):
                _, name_ok, dep_ok, arr_ok = _compare_section(section, ref, country, city, station_keys)
                name_pts += 2 * name_ok
                stop_pts += dep_ok + arr_ok
            score += name_pts + stop_pts
        score = min(10, score)
        if acc:
            score += 10  # a correct route always outranks an incorrect one
        if best is None or score >= best:
            best = score
            result.update(map_score=score, departure_arrival=dep_arr, route_name=name_pts, stops=stop_pts)
    return result


def score_long(answer, meta, metro_data):
    """Score a long-question answer. ``meta`` additionally needs ``question_long`` (the asked format)."""
    country, city = meta['country'], meta['city']
    station_keys = list(metro_data)
    via_mode = via_mode_for_question(meta['question_long'])
    result = {
        'acc': 0,
        'map_score': 0,
        'departure_arrival': 0,
        'route_name': 0,
        'stops': 0,
        'num_via_stop_score': 0,
        'via_stops_score': 0,
    }

    route_data = parse_route_sections(answer, via_mode=via_mode)
    if not route_data:
        return result

    acc = route_is_correct(route_data, meta['station_1'], meta['station_2'], metro_data)
    result['acc'] = acc
    endpoints_ok = _endpoints_match(route_data, meta['station_1'], meta['station_2'])

    best = None
    for plan in _reference_plans(meta.get('routes')):
        score = dep_arr = name_pts = stop_pts = 0
        num_via_score = 0
        via_score = 0
        if endpoints_ok:
            score += 1
            dep_arr += 1
            via_hits = 0
            union, inter = set(), set()
            for section, ref in zip(route_data, plan):
                ref_name, name_ok, dep_ok, arr_ok = _compare_section(section, ref, country, city, station_keys)
                name_pts += 2 * name_ok
                stop_pts += dep_ok + arr_ok

                if via_mode == 'num':
                    pred_n = int(section['Number of Via Stops'])
                    ref_n = int(ref['num_via_stops'])
                    error = abs(pred_n - ref_n)
                    if error == 0:
                        num_via_score += 4
                    elif ref_n != 0:
                        num_via_score += max(0, 4 - (error / ref_n) * 4)
                else:
                    ref_via = {clean_string(reference_stop_name(v, country)) for v in (ref.get('via_stops') or [])}
                    pred_via = section['Via Stops']
                    via_hits += sum(1 for v in pred_via if v in ref_via)
                    if name_ok:
                        union.update(ref_via, pred_via)
                        inter.update(ref_via.intersection(pred_via))

            via_hits = min(10, via_hits)
            num_via_score = min(10, num_via_score)
            iou = len(inter) / len(union) if union else 0
            via_score = (iou * 10 + via_hits) / 2
            score += name_pts + stop_pts + via_score + num_via_score
        score = min(20, score)
        if acc:
            score += 20
        if best is None or score >= best:
            best = score
            result.update(
                map_score=score,
                departure_arrival=dep_arr,
                route_name=name_pts,
                stops=stop_pts,
                num_via_stop_score=num_via_score,
                via_stops_score=via_score,
            )
    return result


# ----------------------------------------------------------------------------- ReasonMap-Plus

_PLUS_ANSWER_FLAGS = ('the final answer is', 'the answer is', 'the correct answer is', 'the answer should be')
_PLUS_CHOICES = {'a': 0, 'b': 1, 'c': 2, 'd': 3}


def _fix_fracs(string):
    substrs = string.split('\\frac')
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += '\\frac'
            if len(substr) > 0 and substr[0] == '{':
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != '{':
                    if len(substr) > 2:
                        new_str += '{' + a + '}{' + b + '}' + substr[2:]
                    else:
                        new_str += '{' + a + '}{' + b + '}'
                else:
                    if len(substr) > 2:
                        new_str += '{' + a + '}' + b + substr[2:]
                    else:
                        new_str += '{' + a + '}' + b
    return new_str


def _fix_a_slash_b(string):
    if len(string.split('/')) != 2:
        return string
    a, b = string.split('/')
    try:
        a = int(a)
        b = int(b)
        assert string == '{}/{}'.format(a, b)
        return '\\frac{' + str(a) + '}{' + str(b) + '}'
    except Exception:
        return string


def _remove_right_units(string):
    return string.split('\\text{ ')[0]


def _fix_sqrt(string):
    if '\\sqrt' not in string:
        return string
    splits = string.split('\\sqrt')
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) > 0 and split[0] != '{':
            new_substr = '\\sqrt{' + split[0] + '}' + split[1:]
        else:
            new_substr = '\\sqrt' + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    string = string.replace('\n', '')
    string = string.replace('\\!', '')
    string = string.replace('\\\\', '\\')
    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')
    string = string.replace('\\left', '')
    string = string.replace('\\right', '')
    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')
    string = string.replace('\\$', '')
    string = string.replace('$', '')
    string = _remove_right_units(string)
    string = string.replace('\\%', '')
    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')
    if len(string) == 0:
        return string
    if string[0] == '.':
        string = '0' + string
    if len(string.split('=')) == 2:
        string = string.split('=')[-1]
    if len(string.split('\\approx')) == 2:
        string = string.split('\\approx')[-1]
    if 'sqrt' in string:
        string = _fix_sqrt(string)
    string = string.replace(' ', '')
    if 'sqrt' in string:
        string = _fix_fracs(string)
    if string == '0.5':
        string = '\\frac{1}{2}'
    string = _fix_a_slash_b(string)
    return string


def find_math_answer(s):
    s = s.lower()
    if '{}' in s:
        s = s.replace('{}', '')

    try:
        pattern = re.compile('oxed{(.*)}', flags=re.S)
        ans = pattern.findall(s)[-1]
    except IndexError:
        ans = s  # no \boxed{...}: use the whole string

    # A closing brace with no opening brace before it: keep everything before it.
    if ans.find('}') != -1 and (ans.find('{') == -1 or ans.find('}') < ans.find('{')):
        ans = ans.split('}')[0]

    ans = ans.split('=')[-1]
    ans = ans.split('\\approx')[-1]

    ans = ans.replace(' ', '').replace('\\,', '').replace('∞', '\\infty')
    ans = ans.replace('+\\infty', '\\infty').replace('\\\\', '\\').replace('\n', '')
    ans = ans.replace('\\text', '').replace('\\mbox', '').replace('bmatrix', 'pmatrix')
    ans = ans.replace('\\left', '').replace('\\right', '').replace('^{\\circ}', '')
    ans = ans.replace('^\\circ', '').replace('{m}^3', '').replace('m^3', '')
    ans = ans.replace('{units}', '').replace('units', '').replace('{km}', '').replace('km', '')

    return _strip_string(ans)


def extract_plus_answer(prediction):
    """Pull the short answer (number / letter / yes-no) out of a ReasonMap-Plus prediction."""
    model_answer = strip_reasoning(prediction)
    if 'oxed{' not in model_answer:
        for flag in _PLUS_ANSWER_FLAGS:
            for variant in (flag, flag.replace('the', 'The')):
                raw = model_answer
                model_answer = model_answer.split(variant)[-1].strip()
                if variant in raw:
                    model_answer = model_answer.split('\n')[0].split('. ')[0]
    elif model_answer.count('oxed{') > 1:
        model_answer = '\\boxed{' + model_answer.split('oxed{')[-1]
    return find_math_answer(model_answer).rstrip('.').lstrip(':').strip()


def plus_is_correct(extracted, ground_truth, q_type):
    """``ground_truth`` is 1 / 0 for TorF, the option index for Counting1 and the count otherwise."""
    ground_truth = int(ground_truth)
    if 'TorF' in q_type:
        return extracted.lower() == ('yes' if ground_truth == 1 else 'no')
    if q_type in ('Counting2', 'Counting3'):
        try:
            return int(extracted) == ground_truth
        except (ValueError, TypeError):
            return False
    if q_type == 'Counting1':
        letter = extracted.lower()
        return letter in _PLUS_CHOICES and _PLUS_CHOICES[letter] == ground_truth
    return False


def plus_groups(q_type):
    """Metric groups a ReasonMap-Plus question type is reported under."""
    ql = str(q_type).lower()
    if ql.startswith('count'):
        return ['Counting']
    if ql == 'torf1':
        return ['TorF', 'TorF1']
    if ql == 'torf2':
        return ['TorF', 'TorF2']
    if ql.startswith('torf'):
        return ['TorF']
    return []
