import html
import logging
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata

from bs4 import BeautifulSoup, NavigableString

from src.core.preprocess.text_postprocess import likely_bad_latex, safe_latex_to_text

def remove_markdown_fences(content):
    content = re.sub(r'^```markdown\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'^```html\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'^```latex\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'```\n?$', '', content, flags=re.MULTILINE)
    return content

# Standardize all consecutive characters
def replace_repeated_chars(input_str):
    input_str = re.sub(r'_{4,}', '____', input_str) # Replace more than 4 consecutive underscores with 4 underscores
    input_str = re.sub(r' {4,}', '    ', input_str)   # Replace more than 4 consecutive spaces with 4 spaces
    return input_str
    # return re.sub(r'([^a-zA-Z0-9])\1{10,}', r'\1\1\1\1', input_str) # For other consecutive symbols (except numbers and letters), replace more than 10 occurrences with 4

# Special Unicode handling
def fullwidth_to_halfwidth(s):
    result = []
    for char in s:
        code = ord(char)
        # Convert full-width space to half-width space
        if code == 0x3000:
            code = 0x0020
        # Convert other full-width characters to half-width
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        result.append(chr(code))
    return ''.join(result)

def find_special_unicode(s):
    special_chars = {}
    for char in s:
        if ord(char) > 127:  # Non-ASCII characters
            # unicode_name = unicodedata.name(char, None)
            unicode_name = unicodedata.category(char)
            special_chars[char] = f'U+{ord(char):04X} ({unicode_name})'
    return special_chars

# # Define dictionary for Unicode character replacements
# unicode_replacements = {
#     "\u00A9": r"$\copyright$",  # Copyright symbol © to latex
#     "\u00AE": r"$^\circledR$",  # Registered trademark ® to latex
#     "\u2122": r"$^\text{TM}$",   # Trademark ™ to latex
#     "\u2018": "'",             # Left single quote to straight quote
#     "\u2019": "'",             # Right single quote to straight quote
#     "\u201C": "\"",            # Left double quote to straight quote
#     "\u201D": "\"",            # Right double quote to straight quote
#     "\u2013": "-",             # En dash to hyphen
#     "\u2014": "-",             # Em dash to hyphen
#     "\u2026": "...",           # Unicode ellipsis to three dots
#     "\u2103": r"$\textdegree C$",  # ℃
#     "\u03B1": r"$\alpha$",         # α
#     "\u03B2": r"$\beta$",          # β
#     "\u03A3": r"$\Sigma$",         # Σ
# }

# # Use regex to replace Unicode characters
# def replace_unicode(match):
#     char = match.group(0)
#     return unicode_replacements.get(char, char)

inline_reg = re.compile(
    r'\$(.*?)\$|'
    r'\\\((.*?)\\\)',
)

CIRCLED_UNICODE_MAP = {chr(0x2460 + i): str(i + 1) for i in range(20)}
CIRCLED_UNICODE_MAP.update({chr(0x24D0 + i): chr(ord('a') + i) for i in range(26)})
CIRCLED_UNICODE_MAP.update({chr(0x24B6 + i): chr(ord('A') + i) for i in range(26)})
TEXTCIRCLED_CMD_RE = re.compile(r'\\textcircled\s*\{\s*([^{}]+?)\s*\}')

def _read_float_env(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)



def replace_textcircle(text):
    """将圆圈数字/字母和 \textcircled{...} 统一归一化为其内部字符，避免文本匹配时一边变成普通字符、一边保留圈号。"""
    if text is None:
        return text

    text = str(text)
    text = TEXTCIRCLED_CMD_RE.sub(lambda m: m.group(1).strip(), text)
    return ''.join(CIRCLED_UNICODE_MAP.get(char, char) for char in text)


def _should_skip_inline_textblock_formula(latex_text):
    latex_text = str(latex_text or '').strip()
    if not latex_text:
        return False

    if len(latex_text) > 128:
        return True

    complex_tokens = (
        r'\begin{',
        r'\end{',
        r'\array',
        r'\matrix',
        r'\tabular',
        r'\cases',
        r'\align',
        r'\\',
        '&',
    )
    if any(token in latex_text for token in complex_tokens):
        return True

    if latex_text.count('{') != latex_text.count('}'):
        return True

    return False

def textblock2unicode(text):
    text = replace_textcircle(text)
    inline_matches = inline_reg.finditer(text)
    removal_positions = []
    for match in inline_matches:
        position = [match.start(), match.end()]
        content = match.group(1) if match.group(1) is not None else match.group(2)
        # print('-------- content-------', content)
        # Remove escape characters \
        clean_content = re.sub(r'\\([\\_&%^])', '', content)

        try:
            if any(char in clean_content for char in r'\^_'):
                if clean_content.endswith('\\'):
                    clean_content += ' '
                if _should_skip_inline_textblock_formula(clean_content):
                    continue
                # inline_array.append(match.group(0))
                unicode_content = safe_latex_to_text(clean_content, fallback=clean_content, latex_type='formula')
                removal_positions.append((position[0], position[1], unicode_content))
        except:
            continue
    
    # Remove inline formulas from original text
    for start, end, unicode_content in sorted(removal_positions, reverse=True):
        text = text[:start] + unicode_content.strip() + text[end:]

    return text

def strip_formula_delimiters(text):
    if not isinstance(text, str):
        return ''
    text = text.strip().strip('\n')
    delimiter_pairs = [
        ('$$', '$$'),
        ('\\[', '\\]'),
        ('\\(', '\\)'),
        ('$', '$'),
    ]
    changed = True
    while text and changed:
        changed = False
        for left, right in delimiter_pairs:
            if text.startswith(left) and text.endswith(right) and len(text) >= len(left) + len(right):
                text = text[len(left):len(text) - len(right)].strip()
                changed = True
                break
    return text


def _strip_command_with_balanced_group(text, command):
    text = str(text or '')
    token = f'{command}' + '{'
    while True:
        start = text.find(token)
        if start < 0:
            break
        index = start + len(token)
        depth = 1
        while index < len(text) and depth > 0:
            if text[index] == '{':
                depth += 1
            elif text[index] == '}':
                depth -= 1
            index += 1
        if depth == 0:
            text = text[:start] + text[index:]
        else:
            text = text[:start] + text[start + len(command):]
            break
    return text


def _strip_command_with_optional_star_and_balanced_group(text, command):
    text = str(text or '')
    search_from = 0
    while True:
        start = text.find(command, search_from)
        if start < 0:
            break

        index = start + len(command)
        if index < len(text) and text[index].isalpha():
            search_from = index
            continue

        while index < len(text) and text[index].isspace():
            index += 1
        if index < len(text) and text[index] == '*':
            index += 1
            while index < len(text) and text[index].isspace():
                index += 1

        if index >= len(text) or text[index] != '{':
            search_from = start + len(command)
            continue

        index += 1
        depth = 1
        while index < len(text) and depth > 0:
            if text[index] == '{':
                depth += 1
            elif text[index] == '}':
                depth -= 1
            index += 1

        if depth == 0:
            text = text[:start] + text[index:]
            search_from = start
        else:
            text = text[:start] + text[start + len(command):]
            search_from = start
    return text


def strip_formula_tags(text):
    text = str(text or '')
    if not text:
        return ''
    text = _strip_command_with_optional_star_and_balanced_group(text, '\\tag')
    text = text.replace('\\notag', '')
    text = text.replace('\\nonumber', '')
    return text


def formula_to_text(text):
    text = replace_textcircle(text)
    text = strip_formula_delimiters(text)
    text = strip_formula_tags(text)
    try:
        return safe_latex_to_text(text, fallback=text, latex_type='formula')
    except Exception:
        return text


def normalized_formula(text):
    text = replace_textcircle(text)
    # Normalize math formulas before matching
    filter_list = ['\\mathbf', '\\mathrm', '\\mathnormal', '\\mathit', '\\mathbb', '\\mathcal', '\\mathscr', '\\mathfrak', '\\mathsf', '\\mathtt', 
                   '\\textbf', '\\text', '\\boldmath', '\\boldsymbol', '\\operatorname', '\\bm',
                   '\\symbfit', '\\mathbfcal', '\\symbf', '\\scriptscriptstyle', '\\notag',
                   '\\setlength', '\\coloneqq', '\\space', '\\thickspace', '\\thinspace', '\\medspace', '\\nobreakspace', '\\negmedspace',
                   '\\quad', '\\qquad', '\\enspace', '\\substackw', ' ', '$$', '\\left', '\\right', '\\displaystyle', '\\text']
                #    '\\left', '\\right', '{', '}', ' ']
    
    text = strip_formula_delimiters(text)
    text = strip_formula_tags(text)
    pattern = re.compile(r"\\\[(.+?)(?<!\\)\\\]|\\\((.+?)(?<!\\)\\\)")
    match = pattern.search(text)

    if match:
        text = (match.group(1) or match.group(2) or '').strip()

    text = _strip_command_with_balanced_group(text, '\\phantom')
    text = re.sub(r'\\[!,;:]', '', text)
    text = re.sub(r'(?<!\\)&', '', text)
    text = text.replace('\\mid', '|')
    text = text.replace('\\vert', '|')
    text = text.replace(r'\{', '')
    text = text.replace(r'\}', '')
    text = text.replace('~', '')
    hspace_pattern = re.compile(r"\\hspace\{.*?\}")
    text = hspace_pattern.sub('', text)
    begin_pattern = re.compile(r"\\begin\{.*?\}")
    text = begin_pattern.sub('', text)
    end_pattern = re.compile(r"\\end\{.*?\}")
    text = end_pattern.sub('', text)
    col_sep = re.compile(r"\\arraycolsep.*?\}")
    text = col_sep.sub('', text)
    text = re.sub(r'\{[lcr| ]+\}', '', text)
    text = text.strip('.')
    
    for filter_text in filter_list:
        text = text.replace(filter_text, '')

    simple_group_pattern = re.compile(r'\{([A-Za-z0-9.+\-]+)\}')
    previous_text = None
    while text != previous_text:
        previous_text = text
        text = simple_group_pattern.sub(r'\1', text)
        text = text.replace('{}', '')
        text = re.sub(r'\{\{+', '{', text)
        text = re.sub(r'}}+', '}', text)
        text = re.sub(
            r'\{([^{}]{1,160})\}',
            lambda m: m.group(1) if any(token in m.group(1) for token in ['=', '+', '-', '\\\\', '^', '_', '[', ']', '(', ')']) else m.group(0),
            text,
        )
        
    # text = normalize_text(delimiter_filter(text))
    # text = delimiter_filter(text)
    text = text.lower()
    text = re.sub(r'\s+', '', text)
    return text


TABLE_LATEX_TEXT_REPLACEMENTS = [
    (r'\checkmark', '√'),
    (r'\checked', '√'),
    (r'\surd', '√'),
    (r'\times', '×'),
    (r'\xmark', '×'),
    (r'\crossmark', '×'),
    (r'\pm', '±'),
    (r'\mp', '∓'),
    (r'\alpha', 'α'),
    (r'\beta', 'β'),
    (r'\gamma', 'γ'),
    (r'\gama', 'γ'),
    (r'\mu', 'μ'),
    (r'\lambda', 'λ'),
    (r'\theta', 'θ'),
    (r'\eta', 'η'),
    (r'\pi', 'π'),
    (r'\rho', 'ρ'),
    (r'\sigma', 'σ'),
    (r'\omega', 'ω'),
    (r'\delta', 'δ'),
    (r'\Delta', 'Δ'),
    (r'\epsilon', 'ε'),
    (r'\varepsilon', 'ε'),
    (r'\phi', 'φ'),
    (r'\varphi', 'φ'),
    (r'\tau', 'τ'),
    (r'\partial', '∂'),
    (r'\varnothing', '∅'),
    (r'\emptyset', '∅'),
    (r'\sim', '～'),
]

TABLE_DIRECT_TEXT_REPLACEMENTS = {
    '✓': '√',
    '✔': '√',
    '☑': '√',
    '✅': '√',
    '🗸': '√',
    '✗': '×',
    '✘': '×',
    '✕': '×',
    '✖': '×',
    '☒': '×',
    '❌': '×',
    '╳': '×',
    '⨯': '×',
    'Ø': '∅',
    '∼': '～',
    '〜': '～',
}

TABLE_STANDALONE_TEXT_REPLACEMENTS = {
    '-': '—',
    '–': '—',
    '−': '—',
    '－': '—',
    '—': '—',
}

TABLE_MATH_WRAPPERS = [
    ('$$', '$$'),
    (r'\[', r'\]'),
    (r'\(', r'\)'),
    ('$', '$'),
]


def _apply_table_symbol_replacements(text):
    text = str(text or '')
    for src, dst in TABLE_LATEX_TEXT_REPLACEMENTS:
        text = text.replace(src, dst)
    for src, dst in TABLE_DIRECT_TEXT_REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = text.replace(r'\%', '%')
    text = text.replace(r'\#', '#')
    text = text.replace(r'\&', '&')
    text = text.replace(r'\_', '_')
    return text


def _strip_simple_table_math_wrappers(text):
    text = str(text or '').strip()
    changed = True
    while text and changed:
        changed = False
        for left, right in TABLE_MATH_WRAPPERS:
            if text.startswith(left) and text.endswith(right) and len(text) >= len(left) + len(right):
                inner = text[len(left):len(text) - len(right)].strip()
                if not inner or len(inner) > 128:
                    continue
                if '<' in inner or '>' in inner:
                    continue
                if inner.count('$') > 0:
                    continue
                text = inner
                changed = True
                break
    return text


def normalize_table_cell_text(text):
    text = unicodedata.normalize('NFKC', str(text or ''))
    text = text.replace('\xa0', ' ').replace('\u200b', '')
    text = _apply_table_symbol_replacements(text)
    text = _strip_simple_table_math_wrappers(text)
    text = _apply_table_symbol_replacements(text)

    stripped = text.strip()
    if stripped in TABLE_STANDALONE_TEXT_REPLACEMENTS:
        text = TABLE_STANDALONE_TEXT_REPLACEMENTS[stripped]

    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_table_html_cell_text(html_content):
    soup = BeautifulSoup(str(html_content or ''), 'html.parser')
    for cell in soup.find_all(['td', 'th']):
        for node in list(cell.descendants):
            if isinstance(node, NavigableString):
                normalized_text = normalize_table_cell_text(str(node))
                if normalized_text != str(node):
                    node.replace_with(normalized_text)
    return str(soup)


def normalized_html_table(text):
    def process_table_html(md_i):
        def _process_table_dom(html_content):
            soup = BeautifulSoup(html_content, 'html.parser')
            th_tags = soup.find_all('th')
            for th in th_tags:
                th.name = 'td'
            thead_tags = soup.find_all('thead')
            for thead in thead_tags:
                thead.unwrap()
            tbody_tags = soup.find_all('tbody')
            for tbody in tbody_tags:
                tbody.unwrap()
            math_tags = soup.find_all('math')
            for math_tag in math_tags:
                alttext = math_tag.get('alttext', '')
                alttext = f'${alttext}$'
                if alttext:
                    math_tag.replace_with(alttext)
            span_tags = soup.find_all('span')
            for span in span_tags:
                span.unwrap()
            return normalize_table_html_cell_text(str(soup))

        table_res = ''
        table_res_no_space = ''
        if '<table' in md_i.replace(' ', '').replace("'", '"'):
            md_i = _process_table_dom(md_i)
            table_res = html.unescape(md_i).replace('\n', '')
            table_res = unicodedata.normalize('NFKC', table_res).strip()
            pattern = r'<table\b[^>]*>(.*)</table>'
            tables = re.findall(pattern, table_res, re.DOTALL | re.IGNORECASE)
            table_res = ''.join(tables)
            table_res = re.sub('( style=".*?")', '', table_res)
            table_res = re.sub('( height=".*?")', '', table_res)
            table_res = re.sub('( width=".*?")', '', table_res)
            table_res = re.sub('( align=".*?")', '', table_res)
            table_res = re.sub('( class=".*?")', '', table_res)
            table_res = re.sub('</?tbody>', '', table_res)

            table_res = re.sub(r'\s+', ' ', table_res)
            table_res_no_space = '<html><body><table border="1" >' + table_res.replace(' ', '') + '</table></body></html>'
            table_res_no_space = re.sub('colspan="', ' colspan="', table_res_no_space)
            table_res_no_space = re.sub('rowspan="', ' rowspan="', table_res_no_space)
            table_res_no_space = re.sub('border="', ' border="', table_res_no_space)

            table_res = '<html><body><table border="1" >' + table_res + '</table></body></html>'

        return table_res, table_res_no_space

    def clean_table(input_str, flag=True):
        if flag:
            input_str = input_str.replace('<sup>', '').replace('</sup>', '')
            input_str = input_str.replace('<sub>', '').replace('</sub>', '')
            input_str = input_str.replace('<span>', '').replace('</span>', '')
            input_str = input_str.replace('<div>', '').replace('</div>', '')
            input_str = input_str.replace('<p>', '').replace('</p>', '')
            input_str = input_str.replace('<spandata-span-identity="">', '')
            input_str = re.sub('<colgroup>.*?</colgroup>', '', input_str)
        return input_str

    norm_text, _ = process_table_html(text)
    norm_text = clean_table(norm_text)
    return norm_text

def normalized_latex_table(text):
    def latex_template(latex_code):  
        template = r'''
        \documentclass[border=20pt]{article}
        \usepackage{subcaption}
        \usepackage{url}
        \usepackage{graphicx}
        \usepackage{caption}
        \usepackage{multirow}
        \usepackage{booktabs}
        \usepackage{color}
        \usepackage{colortbl}
        \usepackage{xcolor,soul,framed}
        \usepackage{fontspec}
        \usepackage{amsmath,amssymb,mathtools,bm,mathrsfs,textcomp}
        \setlength{\parindent}{0pt}''' + \
        r'''
        \begin{document}
        ''' + \
        latex_code + \
        r'''
        \end{document}'''
    
        return template

    def process_table_latex(latex_code):
        SPECIAL_STRINGS= [
            ['\\\\vspace\\{.*?\\}', ''],
            ['\\\\hspace\\{.*?\\}', ''],
            ['\\\\rule\{.*?\\}\\{.*?\\}', ''],
            ['\\\\addlinespace\\[.*?\\]', ''],
            ['\\\\addlinespace', ''],
            ['\\\\renewcommand\\{\\\\arraystretch\\}\\{.*?\\}', ''],
            ['\\\\arraystretch\\{.*?\\}', ''],
            ['\\\\(row|column)?colors?\\{[^}]*\\}(\\{[^}]*\\}){0,2}', ''],
            ['\\\\color\\{.*?\\}', ''],
            ['\\\\textcolor\\{.*?\\}', ''],
            ['\\\\rowcolor(\\[.*?\\])?\\{.*?\\}', ''],
            ['\\\\columncolor(\\[.*?\\])?\\{.*?\\}', ''],
            ['\\\\cellcolor(\\[.*?\\])?\\{.*?\\}', ''],
            ['\\\\colorbox\\{.*?\\}', ''],
            ['\\\\(tiny|scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|Huge)', ''],
            [r'\s+', ' '],
            ['\\\\centering', ''],
            ['\\\\begin\\{table\\}\\[.*?\\]', '\\\\begin{table}'],
            ['\t', ''],
            ['@{}', ''],
            ['\\\\toprule(\\[.*?\\])?', '\\\\hline'],
            ['\\\\bottomrule(\\[.*?\\])?', '\\\\hline'],
            ['\\\\midrule(\\[.*?\\])?', '\\\\hline'],
            ['p\\{[^}]*\\}', 'l'],
            ['m\\{[^}]*\\}', 'c'],
            ['\\\\scalebox\\{[^}]*\\}\\{([^}]*)\\}', '\\1'],
            ['\\\\textbf\\{([^}]*)\\}', '\\1'],
            ['\\\\textit\\{([^}]*)\\}', '\\1'],
            ['\\\\cmidrule(\\[.*?\\])?\\(.*?\\)\\{([0-9]-[0-9])\\}', '\\\\cline{\\2}'],
            ['\\\\hline', ''],
            [r'\\multicolumn\{1\}\{[^}]*\}\{((?:[^{}]|(?:\{[^{}]*\}))*)\}', r'\1']
        ]
        pattern = r'\\begin\{tabular\}.*\\end\{tabular\}'  # 注意这里不用 .*?
        matches = re.findall(pattern, latex_code, re.DOTALL)
        latex_code = ' '.join(matches)

        for special_str in SPECIAL_STRINGS:
            latex_code = re.sub(fr'{special_str[0]}', fr'{special_str[1]}', latex_code)

        return latex_code
    
    def convert_latex_to_html(latex_content):
        latex_content = str(latex_content or '')
        is_bad, _ = likely_bad_latex(latex_content, latex_type='table')
        if is_bad:
            return ''

        timeout_sec = _read_float_env('OMNIDOCBENCH_LATEXMLC_TIMEOUT_SEC', 60.0)
        timeout_sec = timeout_sec if timeout_sec > 0 else None
        cache_dir = tempfile.mkdtemp(prefix='omnidoc_latexml_')
        tex_path = os.path.join(cache_dir, 'table.tex')
        html_path = os.path.join(cache_dir, 'table.html')
        log_path = os.path.join(cache_dir, 'table.log')

        with open(tex_path, 'w') as f:
            f.write(latex_template(latex_content))

        cmd = ['latexmlc', '--quiet', '--nocomments', f'--log={log_path}', tex_path, f'--dest={html_path}']
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_sec,
            )
            with open(html_path, 'r') as f:
                html_content = f.read()

            pattern = r'<table\b[^>]*>(.*)</table>'
            tables = re.findall(pattern, html_content, re.DOTALL | re.IGNORECASE)
            tables = [f'<table>{table}</table>' for table in tables]
            html_content = '\n'.join(tables)
        except subprocess.TimeoutExpired:
            logging.warning(f"latexmlc exceeded {timeout_sec}s, >>>{latex_content[:120]}...<<<")
            html_content = ''
        except Exception as exc:
            logging.warning(f"latexmlc failed: {exc}, >>>{latex_content[:120]}...<<<")
            html_content = ''
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)
        return html_content
    
    html_text = convert_latex_to_html(text)
    normlized_tables = normalized_html_table(html_text)
    return normlized_tables


def normalized_table(text, format='html'):
    if format not in ['html', 'latex']:
        raise ValueError('Invalid format: {}'.format(format))
    else:
        return globals()['normalized_{}_table'.format(format)](text)


def normalized_text(text):
    return clean_string(textblock2unicode(text))


def textblock_with_norm_formula(text):
    inline_matches = inline_reg.finditer(text)
    removal_positions = []
    for match in inline_matches:
        position = [match.start(), match.end()]
        content = match.group(1) if match.group(1) is not None else match.group(2)
        # print('-------- content-------', content)

        norm_content = normalized_formula(content)
        removal_positions.append((position[0], position[1], norm_content))
    
    # Remove inline formulas from original text
    for start, end, norm_content in sorted(removal_positions, reverse=True):
        text = text[:start] + norm_content.strip() + text[end:]

    return text

# def inline_filter_unicode(text):
#     # Ensure text is string type
#     if not isinstance(text, str):
#         text = str(text)
    
#     # Convert LaTeX content to Unicode representation
#     text = LatexNodes2Text().latex_to_text(text)
    
#     inline_array = []
#     inline_matches = inline_reg.finditer(text)
    
#     for match in inline_matches:
#         position = [match.start(), match.end()]
#         content = match.group(1) if match.group(1) is not None else match.group(2)
        
#         # Remove escape characters \
#         clean_content = re.sub(r'\\([\\_&%^])', '', content)

#         if any(char in clean_content for char in r'\^_'):
#             # inline_array.append(match.group(0))
#             inline_array.append({
#                 'category_type': 'equation_inline',
#                 'position': position,
#                 'content': match.group(0),
#             })
#             text = text.replace(match.group(0), '')
#             # print('-----Found inline formula: ', match.group(0))
#         else:
#             text = text.replace(match.group(0), content)
#         # # Add to inline_array
#         # inline_array.append({
#         #     'category_type': 'equation_inline',
#         #     'position': position,
#         #     'content': content,
#         # })
        
#         # # Remove matched formula from original text, can choose to replace with spaces or remove directly
#         # text = text[:position[0]] + ' '*(position[1]-position[0]) + text[position[1]:]

#     return text, inline_array

def inline_filter_unicode(text):
    # Ensure text is string type
    if not isinstance(text, str):
        text = str(text)
    
    # Replace inline formula boundary markers
    #print('--------text-------',text)
    placeholder = '__INLINE_FORMULA_BOUNDARY__'
    text_copy = text.replace('$', placeholder).replace('\\(', placeholder).replace('\\)', placeholder)
    #print('--------text_copy-------',text_copy)
    # Convert LaTeX content to Unicode representation
    text_copy = safe_latex_to_text(text_copy, fallback=text_copy, latex_type='text')
    #print('--------text_copy---unicode----',text_copy)
    # Restore boundary markers
    text_copy = text_copy.replace(placeholder, '$')
    
    inline_array = []
    inline_matches = inline_reg.finditer(text_copy)
    # Record positions of inline formulas to be removed
    removal_positions = []
    
    for match in inline_matches:
        position = [match.start(), match.end()]
        content = match.group(1) if match.group(1) is not None else match.group(2)
        print('-------- content-------', content)
        # Remove escape characters \
        clean_content = re.sub(r'\\([\\_&%^])', '', content)

        if any(char in clean_content for char in r'\^_'):
            # inline_array.append(match.group(0))
            inline_array.append({
                'category_type': 'equation_inline',
                'position': position,
                'content': content,
            })
            removal_positions.append((position[0], position[1]))
    
    # Remove inline formulas from original text
    for start, end in sorted(removal_positions, reverse=True):
        text = text[:start] + text[end:]

    return text, inline_array

def inline_filter(text):
    # Ensure text is string type
    if not isinstance(text, str):
        text = str(text)
    
    inline_array = []
    inline_matches = inline_reg.finditer(text)
    
    for match in inline_matches:
        position = [match.start(), match.end()]
        content = match.group(1) if match.group(1) is not None else match.group(2)
        # print('inline_content: ', content)
        
        # Remove escape characters \
        clean_content = re.sub(r'\\([\\_&%^])', '', content)

        if any(char in clean_content for char in r'\^_'):
            # inline_array.append(match.group(0))
            inline_array.append({
                'category_type': 'equation_inline',
                'position': position,
                'content': match.group(0),
            })
            text = text.replace(match.group(0), '')
            # print('-----Found inline formula: ', match.group(0))
        else:
            text = text.replace(match.group(0), content)

    return text, inline_array

# Text OCR quality check processing:
def clean_string(input_string):
    input_string = replace_textcircle(input_string)
    # Use regex to keep Chinese characters, English letters and numbers
    # input_string = input_string.replace('\\t', '').replace('\\n', '').replace('\t', '').replace('\n', '').replace('/t', '').replace('/n', '')
    input_string = input_string.replace('\\t', '').replace('\\n', '').replace('\t', '').replace('\n', '').replace('/t', '').replace('/n', '')
    cleaned_string = re.sub(r'[^\w\u4e00-\u9fff]', '', input_string)   # 只保留中英文和数字
    return cleaned_string
