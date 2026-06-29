import html
import re
import subprocess
import unicodedata
import uuid

from bs4 import BeautifulSoup

from src.core.preprocess.data_preprocess import normalize_table_cell_text
from src.core.preprocess.text_postprocess import text_post_process


def _normalize_table_dom(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    for th in soup.find_all('th'):
        th.name = 'td'

    for thead in soup.find_all('thead'):
        thead.unwrap()

    for tbody in soup.find_all('tbody'):
        tbody.unwrap()

    for math_tag in soup.find_all('math'):
        alttext = math_tag.get('alttext', '')
        if alttext:
            math_tag.replace_with(f'${alttext}$')

    for span in soup.find_all('span'):
        span.unwrap()

    return str(soup)


def _clean_table_tags(input_str, flag=True):
    if flag:
        input_str = input_str.replace('<sup>', '').replace('</sup>', '')
        input_str = input_str.replace('<sub>', '').replace('</sub>', '')
        input_str = input_str.replace('<span>', '').replace('</span>', '')
        input_str = input_str.replace('<div>', '').replace('</div>', '')
        input_str = input_str.replace('<p>', '').replace('</p>', '')
        input_str = input_str.replace('<spandata-span-identity="">', '')
        input_str = re.sub('<colgroup>.*?</colgroup>', '', input_str)
    return input_str


def table_structure_post_process(text):
    table_text = str(text or '')
    if '<table' not in table_text.replace(' ', '').replace("'", '"'):
        return ''

    table_text = _normalize_table_dom(table_text)
    table_text = html.unescape(table_text).replace('\n', '')
    table_text = unicodedata.normalize('NFKC', table_text).strip()

    pattern = r'<table\b[^>]*>(.*)</table>'
    tables = re.findall(pattern, table_text, re.DOTALL | re.IGNORECASE)
    table_text = ''.join(tables)
    table_text = re.sub('( style=".*?")', '', table_text)
    table_text = re.sub('( height=".*?")', '', table_text)
    table_text = re.sub('( width=".*?")', '', table_text)
    table_text = re.sub('( align=".*?")', '', table_text)
    table_text = re.sub('( class=".*?")', '', table_text)
    table_text = re.sub('</?tbody>', '', table_text)
    table_text = re.sub(r'\s+', ' ', table_text)
    table_text = '<html><body><table border="1" >' + table_text + '</table></body></html>'
    table_text = _clean_table_tags(table_text)
    return table_text.replace('> ', '>').replace(' </td>', '</td>')


def table_content_post_process(html_text):
    html_text = str(html_text or '')
    if not html_text:
        return html_text

    pattern = r'(<td[^>]*>)(.*?)(</td>)'

    def replace_func(match):
        start_tag = match.group(1)
        content = match.group(2)
        end_tag = match.group(3)
        return start_tag + normalize_table_cell_text(text_post_process(content)) + end_tag

    return re.sub(pattern, replace_func, html_text, flags=re.DOTALL)


def table_post_process(table_str):
    table_str = table_structure_post_process(table_str)
    table_str = table_content_post_process(table_str)
    return table_str


def table_to_text_lines(table_str):
    table_html = table_post_process(table_str)
    if not table_html:
        return []

    soup = BeautifulSoup(table_html, 'html.parser')
    text_lines = []

    for row in soup.find_all('tr'):
        row_cells = []
        for cell in row.find_all('td'):
            cell_text = normalize_table_cell_text(text_post_process(cell.get_text(' ', strip=True)))
            if cell_text:
                row_cells.append(cell_text)
        if row_cells:
            text_lines.append(' '.join(row_cells))

    if text_lines:
        return text_lines

    fallback_cells = []
    for cell in soup.find_all('td'):
        cell_text = normalize_table_cell_text(text_post_process(cell.get_text(' ', strip=True)))
        if cell_text:
            fallback_cells.append(cell_text)
    if fallback_cells:
        return fallback_cells

    fallback_text = normalize_table_cell_text(text_post_process(soup.get_text(' ', strip=True)))
    return [fallback_text] if fallback_text else []
