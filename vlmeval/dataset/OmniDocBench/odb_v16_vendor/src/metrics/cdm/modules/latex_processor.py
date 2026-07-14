import os
import re
import json
import shutil
import logging
import numpy as np
from PIL import Image


SKIP_PATTERNS = [r'\{', r'\}', r'[\[\]]', r'\\begin\{.*?\}', r'\\end\{.*?\}', r'\^', r'\_', r'\\.*rule.*', r'\\.*line.*', r'\[[\-.0-9]+[epm][xtm]\]']
SKIP_Tokens = ['\\', '\\\\', '\\index', '\\a', '&', '$', '\\multirow', '\\def', '\\edef', '\\raggedright', '\\url', '\\cr', '\\ensuremath', '\\left', '\\right', 
               '\\mathchoice', '\\scriptstyle', '\\displaystyle', '\\qquad', '\\quad', '\\,', '\\!', '~', '\\boldmath', '\\gdef', '\\today', '\\the']
PHANTOM_Tokens = ['\\fontfamily', '\\vphantom', '\\phantom', '\\rowcolor', '\\ref', '\\thesubequation', '\\global', '\\theboldgroup']
TWO_Tail_Tokens = ['\\frac', '\\binom']
AB_Tail_Tokens = ['\\xrightarrow', '\\xleftarrow', '\\sqrt']        # special token \xxx [] {} 
TWO_Tail_Invisb_Tokens = ['\\overset', '\\underset', '\\stackrel']
ONE_Tail_Tokens = ['\\widetilde', '\\overline', '\\hat', '\\widehat', '\\tilde', '\\Tilde', '\\dot', '\\bar', '\\vec', '\\underline', '\\underbrace', '\\check',
                   '\\breve', '\\Bar', '\\Vec', '\\mathring', '\\ddot', '\\Ddot', '\\dddot', '\\ddddot']
ONE_Tail_Invisb_Tokens = ['\\boldsymbol', '\\pmb', '\\textbf', '\\mathrm', '\\mathbf', '\\mathbb', '\\mathcal', '\\textmd', '\\texttt', '\\textnormal', 
                          '\\text', '\\textit', '\\textup', '\\mathop', '\\mathbin', '\\smash', '\\operatorname', '\\textrm', '\\mathfrak', '\\emph',
                          '\\textsf', '\\textsc']


def flatten_multiline(latex):
    brace_map = {
        "\\left(": "\\right)",
        "\\left[": "\\right]",
        "\\left{": "\\right}",
    }
    l_split = latex.split(' ')
    if l_split[0] == "\\begin{array}":
        if l_split[-1] == "\\end{array}":
            l_split = l_split[2:-1]
        else:
            l_split = l_split[2:]
    
    idx = 0
    while idx < len(l_split):
        token = l_split[idx]
        if token.startswith("\\left") and token in brace_map.keys():
            end_idx = find_matching_brace(l_split, idx, brace=[token, brace_map[token]])
            if end_idx != -1:
                idx = end_idx
        elif token in ["\\\\", "~", "\\qquad"]:
            l_split = l_split[0:idx] + l_split[idx+1:]
            idx -= 1
        idx += 1
    latex = ' '.join(l_split)
    return "$ "+latex+" $"
    
    
def clean_latex(text):
    # 去除非转义字符之间的多余空格
    cleaned_text = re.sub(r'(?<=[^\\])\s+(?=[^\\])', '', text)
    # 补回关键命令后必须保留的空格
    for item in ["\\hline", "\\midrule", "\\times", "\\bf", "\\footnotesize", "\\cr", '\\log']:
        cleaned_text = cleaned_text.replace(item, item+" ")
    cleaned_text = cleaned_text.replace(" \\mathcolor{black}", "\\mathcolor{black}")
    return cleaned_text

def remove_trailing_latex(formula):
    pattern = r'(\\(hspace\*?\{[^{}]*?\}|vspace\*?\{[^{}]*?\}|smallskip|medskip|quad|qquad|bigskip|[;,])|\~|\.)*$'
    # Replace the matched pattern with an empty string
    cleaned_formula = re.sub(pattern, '', formula, count=1)
    return cleaned_formula

def find_matching_brace(sequence, start_index, brace=['{', '}']):
    # Finds the index of the matching brace for the one at start_index
    left_brace, right_brace = brace
    depth = 0
    for i, char in enumerate(sequence[start_index:], start=start_index):
        if char == left_brace:
            depth += 1
        elif char == right_brace:
            depth -= 1
            if depth == 0:
                return i
    if depth > 0:
        error_info = "Warning! found no matching brace in sequence !"
        raise ValueError(error_info)
    return -1

def normalize_latex(l, rm_trail=False):
    if "tabular" in l:
        latex_type = "tabular"
    else:
        latex_type = "formula"
        
    if rm_trail:
        l = remove_trailing_latex(l)
    l = l.strip().replace(r'\pmatrix', r'\mypmatrix').replace(r'\matrix', r'\mymatrix')
    
    # 移除对齐方式命令（\raggedright, \arraybackslash），渲染时难以处理
    for item in ['\\raggedright', '\\arraybackslash']:
        l = l.replace(item, "")
    
    for item in ['\\lowercase', '\\uppercase']:
        l = l.replace(item, "")
        
    # 处理 \hspace/\vspace：公式中去掉空格，表格中直接移除
    pattern = r'\\[hv]space { [.0-9a-z ]+ }'
    old_token = re.findall(pattern, l, re.DOTALL)
    if latex_type == "tabular":
        new_token = ["" for item in old_token]
    else:
        new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft)
        
    # 将 \begin{tabular} 及 \begin{array} 与其参数合并为一个 token（公式和表格均需处理）
    if latex_type == "tabular":
        l = l.replace("\\begin {tabular}", "\\begin{tabular}")
        l = l.replace("\\end {tabular}", "\\end{tabular}")
        l = l.replace("\\begin {array}", "\\begin{array}")
        l = l.replace("\\end {array}", "\\end{array}")
        l_split = l.split(' ')
        idx = 0
        while idx < len(l_split):
            token = l_split[idx]
            if token == "\\begin{tabular}":
                sub_idx = idx + 1
                end_idx = find_matching_brace(l_split, sub_idx)
                new_token = "".join(l_split[idx: end_idx+1])
                l_split = l_split[0:idx] + [new_token] + l_split[end_idx+1:]
                break
            idx += 1
        l = ' '.join(l_split)
        
        # 复杂命令（如 \cmidrule）使用括号匹配而非正则来合并 token
        l_split = l.split(' ')
        idx = 0
        while idx < len(l_split):
            token = l_split[idx]
            if token in ["\\cmidrule", "\\cline"]:
                sub_idx = idx + 1
                if l_split[sub_idx] == "(":
                    mid_end = find_matching_brace(l_split, sub_idx, brace=['(', ')'])
                    end_idx = find_matching_brace(l_split, mid_end+1)
                else:
                    end_idx = find_matching_brace(l_split, sub_idx)
                new_token = "".join(l_split[idx: end_idx+1])
                l_split = l_split[0:idx] + [new_token] + l_split[end_idx+1:]
            idx += 1
        l = ' '.join(l_split)
    
    pattern = r'\\begin{array} { [lrc ]+ }'
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace("\\begin{array} ", "<s>").replace(" ", "").replace("<s>", "\\begin{array} ") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft)
    
    # 将省略号及数学函数名拆分为字符，以便 bbox 匹配
    
    l = " "+l+" "
    l = re.sub(r'(?<=\s)--(?=\s)', r'- -', l)
    l = re.sub(r'(?<=\s)---(?=\s)', r'- - -', l)
    l = re.sub(r'(?<=\s)…(?=\s)', r'. . .', l)
    l = re.sub(r'(?<=\s)\\ldots(?=\s)', r'. . .', l)
    l = re.sub(r'(?<=\s)\\hdots(?=\s)', r'. . .', l)
    l = re.sub(r'(?<=\s)\\cdots(?=\s)', r'. . .', l)
    l = re.sub(r'(?<=\s)\\dddot(?=\s)', r'. . .', l)
    l = re.sub(r'(?<=\s)\\dots(?=\s)', r'. . .', l)
    l = re.sub(r'(?<=\s)\\dotsc(?=\s)', r'. . .', l)
    l = re.sub(r'(?<=\s)\\dotsi(?=\s)', r'. . .', l)
    l = re.sub(r'(?<=\s)\\dotsm(?=\s)', r'. . .', l)
    l = re.sub(r'(?<=\s)\\dotso(?=\s)', r'. . .', l)
    l = re.sub(r'(?<=\s)\\dotsb(?=\s)', r'. . .', l)
    l = re.sub(r'(?<=\s)\\mathellipsis(?=\s)', r'. . .', l)
    l = re.sub(r'(?<=\s)\\ex(?=\s)', r'\\mathrm { e x }', l)
    l = re.sub(r'(?<=\s)\\ln(?=\s)', r'\\mathrm { l n }', l)
    l = re.sub(r'(?<=\s)\\lg(?=\s)', r'\\mathrm { l g }', l)
    l = re.sub(r'(?<=\s)\\cot(?=\s)', r'\\mathrm { c o t }', l)
    l = re.sub(r'(?<=\s)\\mod(?=\s)', r'\\mathrm { m o d }', l)
    l = re.sub(r'(?<=\s)\\bmod(?=\s)', r'\\mathrm { m o d }', l)
    l = re.sub(r'(?<=\s)\\pmod(?=\s)', r'\\mathrm { m o d }', l)  # \pmod 其实和mod不一样，但是不太好处理，暂时替换为\mod
    l = re.sub(r'(?<=\s)\\min(?=\s)', r'\\mathrm { m i n }', l) 
    l = re.sub(r'(?<=\s)\\max(?=\s)', r'\\mathrm { m a x }', l) 
    l = re.sub(r'(?<=\s)\\ker(?=\s)', r'\\mathrm { k e r }', l) 
    l = re.sub(r'(?<=\s)\\hom(?=\s)', r'\\mathrm { h o m }', l)
    l = re.sub(r'(?<=\s)\\sec(?=\s)', r'\\mathrm { s e c }', l)
    l = re.sub(r'(?<=\s)\\scs(?=\s)', r'\\mathrm { s c s }', l)
    l = re.sub(r'(?<=\s)\\csc(?=\s)', r'\\mathrm { c s c }', l)
    l = re.sub(r'(?<=\s)\\deg(?=\s)', r'\\mathrm { d e g }', l)
    l = re.sub(r'(?<=\s)\\arg(?=\s)', r'\\mathrm { a r g }', l)
    l = re.sub(r'(?<=\s)\\log(?=\s)', r'\\mathrm { l o g }', l)
    l = re.sub(r'(?<=\s)\\dim(?=\s)', r'\\mathrm { d i m }', l)
    l = re.sub(r'(?<=\s)\\exp(?=\s)', r'\\mathrm { e x p }', l)
    l = re.sub(r'(?<=\s)\\sin(?=\s)', r'\\mathrm { s i n }', l)
    l = re.sub(r'(?<=\s)\\cos(?=\s)', r'\\mathrm { c o s }', l)
    l = re.sub(r'(?<=\s)\\tan(?=\s)', r'\\mathrm { t a n }', l)
    l = re.sub(r'(?<=\s)\\tanh(?=\s)', r'\\mathrm { t a n h }', l)
    l = re.sub(r'(?<=\s)\\cosh(?=\s)', r'\\mathrm { c o s h }', l)
    l = re.sub(r'(?<=\s)\\sinh(?=\s)', r'\\mathrm { s i n h }', l)
    l = re.sub(r'(?<=\s)\\coth(?=\s)', r'\\mathrm { c o t h }', l)
    l = re.sub(r'(?<=\s)\\arcsin(?=\s)', r'\\mathrm { a r c s i n }', l)
    l = re.sub(r'(?<=\s)\\arccos(?=\s)', r'\\mathrm { a r c c o s }', l)
    l = re.sub(r'(?<=\s)\\arctan(?=\s)', r'\\mathrm { a r c t a n }', l)
    
    # ** token such as \string xxx should be one token
    pattern = r'\\string [^ ]+ '
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft+" ")
    
    # ** token such as \big( should be one token
    pattern = r'\\[Bb]ig[g]?[glrm]? [(){}|\[\]] '
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft+" ")
        
    pattern = r'\\[Bb]ig[g]?[glrm]? \\.*? '
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft+" ")
        
    # \operatorname * 与 mathcolor 冲突，移除多余的 *
    pattern = r'\\operatorname \*'
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = ["\\operatorname" for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft)
    
    # \lefteqn 会导致字符重叠，直接移除
    l = l.replace("\\lefteqn", "")
    
    # \footnote 无法按常规方式着色，替换为上标符号 "^"
    l = l.replace("\\footnote ", "^ ")
    
    # \' 单独渲染会产生视觉差异，将 \' e 合并为 \'e（后跟 { 时除外）
    pattern = r'\\\' [^{] '
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft+" ")
    
    # 表格中的排版微调参数（如 [-1.5ex]）合并为一个 token，无需渲染
    if latex_type == "tabular":
        pattern = r'\[ [\-.0-9 ]+[exptcm ]+ \]'
        old_token = re.findall(pattern, l, re.DOTALL)
        new_token = [item.replace(" ", "") for item in old_token]
        for bef, aft in zip(old_token, new_token):
            l = l.replace(bef, aft)
    
    # ** \parbox { 3cm } {} shoudle be combined as one token
    pattern = r'\\parbox {[^{]+}'
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft)
    
    # ** \raisebox{<lift>}[<height>][<depth>] {} shoudle be combined as one token, \raisebox{-1.5ex}[0pt]
    pattern = r'\\raisebox {[^{]+} [\[\]0-9 exptcm]+{'
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft[0:-1]+" {")
        
    # ** \char shoudle be combined as one token
    pattern = r'{ \\char[0-9\' ]+}'
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, "{ "+aft[1:-1]+" }")
        
    # ** \rule{1pt}{2pt} lines, shoudle be combined as one token and do not render
    pattern = r'\\rule {[ .0-9a-z]+} {[ .0-9a-z]+}'
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft)
        
    # ** \specialrule{1pt}{2pt}{2pt}, special lines, shoudle be combined as one token
    pattern = r'\\specialrule {[ .0-9a-z]+} {[ .0-9a-z]+} {[ .0-9a-z]+}'
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft)
        
    # ** for easier add color, the original color should be removed, there are two type of color for now: \color[rgb]{0, 1, 0} and \color{red}
    pattern = r'\\colorbox[ \[\]RGBrgb]+{ [A-Za-z 0-9,!]+ } |\\color[ \[\]RGBrgb]+{ [A-Za-z 0-9,!]+ } |\\textcolor[ \[\]RGBrgb]+{ [A-Za-z 0-9,!]+ } |\\cellcolor[ \[\]RGBrgb]+{ [A-Za-z 0-9,!]+ } '
    old_token = re.findall(pattern, l, re.DOTALL)
    for bef in old_token:
        l = l.replace(bef, "")
    
    # ** filling the missing brace [] and {} according to token.
    l_split = l.split(' ')
    idx = 0
    while idx < len(l_split):
        token = l_split[idx]
        if token in ONE_Tail_Tokens + ONE_Tail_Invisb_Tokens:
        # ** normalize tokens such as \hat, fill missing the {}, such as \hat \lambda -> \hat {\lambda}
            sub_idx = idx + 1
            while sub_idx < len(l_split) and l_split[sub_idx] in ONE_Tail_Tokens+ONE_Tail_Invisb_Tokens:
                sub_idx += 1
            new_split = l_split[0:idx]
            for ii in range(idx, sub_idx):
                new_split = new_split + [l_split[ii], "{"]
            if l_split[sub_idx] != "{":
                new_split = new_split + [l_split[sub_idx]] + ["}"]*(sub_idx-idx)
                l_split = new_split + l_split[sub_idx+1:]
            else:
                end_idx = find_matching_brace(l_split, sub_idx)
                new_split = new_split + l_split[sub_idx+1:end_idx] + ["}"]*(sub_idx-idx)
                l_split = new_split + l_split[end_idx+1:]
        elif token in AB_Tail_Tokens:
        # ** normalize special tokens such as \sqrt, fill the missing [] {} in \sqrt [] {}, yet the [] is optional, for example: \sqrt A B -> \sqrt {A} B and \sqrt [A] B -> \sqrt [A] {B}
            if l_split[idx + 1] != "[" and l_split[idx + 1] != "{":
                l_split = l_split[0:idx+1] + ["{"] + [l_split[idx+1]] + ["}"] + l_split[idx+2:]
            else:
                if l_split[idx + 1] == "[":
                    end1 = find_matching_brace(l_split, idx+1, brace=['[', ']'])
                else:
                    end1 = idx
                if l_split[end1 + 1] != "{":
                    l_split = l_split[0:end1+1] + ["{"] + [l_split[end1+1]] + ["}"] + l_split[end1+2:]
        elif token in TWO_Tail_Tokens + TWO_Tail_Invisb_Tokens:
        # ** normalize special tokens such as \frac, add missing brace in \frac {A} {B} for example: \frac {\lambda} 2 -> \frac {\lambda} {2}
            if l_split[idx + 1] != "{":
                l_split = l_split[0:idx+1] + ["{"] + [l_split[idx+1]] + ["}"] + l_split[idx+2:]
            end1 = find_matching_brace(l_split, idx+1)
            if l_split[end1 + 1] != "{":
                l_split = l_split[0:end1+1] + ["{"] + [l_split[end1+1]] + ["}"] + l_split[end1+2:]
            
        idx += 1
    l = ' '.join(l_split)
    
    return l

def token_add_color(l_split, idx, render_dict):
    token = l_split[idx]
    if token in PHANTOM_Tokens:
        # ** special tokens that do not need render, skip it 
        if l_split[idx + 1] == '{':
            brace_end = find_matching_brace(l_split, idx + 1)
        else:
            brace_end = idx + 1
        next_idx = brace_end + 1
    elif token in TWO_Tail_Tokens:
        # ** tokens such as \frac A B, and the token needs render too.
        num_start = idx + 1
        num_end = find_matching_brace(l_split, num_start)
        den_start = num_end + 1
        den_end = find_matching_brace(l_split, den_start)
        l_split_copy = l_split[:idx] + [r'\mathcolor{black}{'+token+'{'] + \
                        [r'\mathcolor{gray}{'] + l_split[num_start + 1:num_end] + \
                        ['}'] + [r'}{'] + [r'\mathcolor{gray}{'] + l_split[den_start + 1:den_end] + \
                        ['}'] + ['}'] + ['}'] + l_split[den_end + 1:]
                        
        l_new = ' '.join(l_split_copy)
        l_new = r'\mathcolor{gray}{ ' + l_new + ' }'
        render_dict[str(idx)] = l_new, token
        next_idx = idx + 1
    elif token in ONE_Tail_Tokens:
        # ** tokens such as \hat A, and the token needs render too.
        num_start = idx + 1
        num_end = find_matching_brace(l_split, num_start)
        l_split_copy = l_split[:idx] + [r'\mathcolor{black}{'] + l_split[idx: num_start+1] + \
                        [r'\mathcolor{gray}{'] + l_split[num_start+1: num_end] + \
                        ['}'] + l_split[num_end: num_end+1] + ['}'] + l_split[num_end+1:]
        l_new = ' '.join(l_split_copy)
        l_new = r'\mathcolor{gray}{ ' + l_new + ' }'
        render_dict[str(idx)] = l_new, token
        next_idx = idx + 1
    elif token in ONE_Tail_Invisb_Tokens:
        # ** tokens such as \text A B, and the token does not need render.
        num_start = idx + 1
        num_end = find_matching_brace(l_split, num_start)
        sub_idx = num_start+1
        if num_end-num_start == 2:
            l_split_copy = l_split.copy()
            l_split_copy[sub_idx] = r'{\mathcolor{black}{' + l_split_copy[sub_idx] + '}}'
            l_new = ' '.join(l_split_copy)
            l_new = r'\mathcolor{gray}{ ' + l_new + ' }'
            render_dict[str(idx)] = l_new, l_split[sub_idx]
            next_idx = num_end
        else:
            while sub_idx < num_end:
                l_split, sub_idx, render_dict = token_add_color(l_split, sub_idx, render_dict)
        next_idx = num_end + 1
    elif token in AB_Tail_Tokens:
        # ** special token \xrightarrow, could be \xrightarrow [] {} or \xrightarrow {}, process method are different.
        if l_split[idx+1] == '{':
            num_start = idx + 1
            num_end = find_matching_brace(l_split, num_start)
            l_split_copy = l_split[:idx] + [r'\mathcolor{black}{'] + l_split[idx: idx+2] \
                        + [r'\mathcolor{gray}{'] + l_split[num_start+1: num_end] + ['}}'] + l_split[num_end:]
            l_new = ' '.join(l_split_copy)
            l_new = r'\mathcolor{gray}{ ' + l_new + ' }'
            render_dict[str(idx)] = l_new, token
            sub_idx = num_start+1
            while sub_idx < num_end:
                l_split, sub_idx, render_dict = token_add_color(l_split, sub_idx, render_dict)
            next_idx = num_end + 1
        elif l_split[idx+1] == '[':
            num_start = idx + 1
            num_end = find_matching_brace(l_split, num_start, brace=['[', ']'])
            den_start = num_end + 1
            den_end = find_matching_brace(l_split, den_start)
            l_split_copy = l_split[:idx] + [r'{\mathcolor{black}{'] + l_split[idx: idx+2] \
                        + [r'\mathcolor{gray}{'] + l_split[idx+2: num_end] + ['}'] + l_split[num_end:den_start+1] \
                        + [r'\mathcolor{gray}{'] + l_split[den_start+1: den_end] + ['}'] + l_split[den_end: den_end+1] \
                        + ['}}'] + l_split[den_end+1:]
            l_new = ' '.join(l_split_copy)
            l_new = r'\mathcolor{gray}{ ' + l_new + ' }'
            render_dict[str(idx)] = l_new, token
            sub_idx = num_start + 1
            while sub_idx < num_end:
                l_split, sub_idx, render_dict = token_add_color(l_split, sub_idx, render_dict)
            sub_idx = den_start + 1
            while sub_idx < den_end:
                l_split, sub_idx, render_dict = token_add_color(l_split, sub_idx, render_dict)
            next_idx = den_end + 1
    elif token in ["\\multicolumn", "\\multirow"]:
        # ** tokens with three {}, such as \multicolumn {} {} {}, the text in third {} need be rendered.
        first_start = idx + 1
        first_end = find_matching_brace(l_split, first_start)
        second_start = first_end + 1
        second_end = find_matching_brace(l_split, second_start)
        third_start = second_end + 1
        third_end = find_matching_brace(l_split, third_start)
        
        sub_idx = third_start+1
        while sub_idx < third_end:
            l_split, sub_idx, render_dict = token_add_color(l_split, sub_idx, render_dict)
        next_idx = third_end + 1
    elif token in SKIP_Tokens+TWO_Tail_Invisb_Tokens or any(re.match(pattern, token) for pattern in SKIP_PATTERNS):
        # ** tokens no need render, just skip
        # 特殊情况：[] 可能是独立方括号，也可能属于 \sqrt[]{}
        if (token == "[" and l_split[idx-1]!="\\sqrt") or (token == "]" and idx>=3 and l_split[idx-3]!="\\sqrt"):
            l_split_copy = l_split.copy()
            l_split_copy[idx] = r'\mathcolor{black}{ ' + l_split_copy[idx] + ' }'
            l_new = ' '.join(l_split_copy)
            l_new = r'\mathcolor{gray}{ ' + l_new + ' }'
            render_dict[str(idx)] = l_new, token
            next_idx = idx + 1
        else:
            next_idx = idx + 1
    else:
        # ** nomal token
        l_split_copy = l_split.copy()
        # 着色时在花括号内保留空格可避免渲染偏移
        l_split_copy[idx] = r'\mathcolor{black}{ ' + l_split_copy[idx] + ' }'

        l_new = ' '.join(l_split_copy)
        l_new = r'\mathcolor{gray}{ ' + l_new + ' }'
        render_dict[str(idx)] = l_new, token
        next_idx = idx + 1
        
    return l_split, next_idx, render_dict


def token_add_color_RGB(l_split, idx, token_list, brace_color=False):
    """using \mathcolor[RGB]{r,g,b} to render latex. 
    """
    token = l_split[idx]
    if not token:
        next_idx = idx + 1
    elif token in PHANTOM_Tokens:
        # ** special tokens that do not need render, skip it 
        if l_split[idx + 1] == '{':
            brace_end = find_matching_brace(l_split, idx + 1)
        else:
            brace_end = idx + 1
        next_idx = brace_end + 1
    elif token in TWO_Tail_Tokens:
        # ** tokens such as \frac A B, and the token needs render too.
        num_start = idx + 1
        num_end = find_matching_brace(l_split, num_start)
        den_start = num_end + 1
        den_end = find_matching_brace(l_split, den_start)
        color_token = "\\mathcolor[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
        l_split = l_split[:idx] + [color_token+token] + l_split[idx+1: den_end+1] + ["}"] + l_split[den_end+1:]
        token_list.append(token)
        next_idx = idx + 1
    elif token in ONE_Tail_Tokens:
        # ** tokens such as \hat A, and the token needs render too.
        num_start = idx + 1
        num_end = find_matching_brace(l_split, num_start)
        color_token = "\\mathcolor[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
        if token != "\\underbrace" and num_end+1 < len(l_split) and l_split[num_end+1] == "_":
            l_split = l_split[:idx] + ["{"+color_token+token] + l_split[idx+1: num_end+1] + ["}}"] + l_split[num_end+1:]
        else:
            l_split = l_split[:idx] + [color_token+token] + l_split[idx+1: num_end+1] + ["}"] + l_split[num_end+1:]
        token_list.append(token)
        next_idx = idx + 1
    elif token in ONE_Tail_Invisb_Tokens:
        # ** tokens such as \text A B, and the token does not need render.
        num_start = idx + 1
        num_end = find_matching_brace(l_split, num_start)
        sub_idx = num_start+1
        if num_end-num_start == 2:
            color_token = "\\mathcolor[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
            token_list.append(l_split[num_start+1])
            l_split = l_split[:num_start+1] + [color_token+l_split[num_start+1]+"}"] + l_split[num_end:]
        else:
            while sub_idx < num_end:
                l_split, sub_idx, token_list = token_add_color_RGB(l_split, sub_idx, token_list)
        next_idx = num_end + 1
    elif token in AB_Tail_Tokens:
        # ** special token \xrightarrow, could be \xrightarrow [] {} or \xrightarrow {}, process method are different.
        if l_split[idx+1] == '{':
            num_start = idx + 1
            num_end = find_matching_brace(l_split, num_start)
            color_token = "\\mathcolor[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
            l_split = l_split[:idx] + [color_token+token] + l_split[idx+1: num_end+1] + ["}"] + l_split[num_end+1:]
            token_list.append(token)
            sub_idx = num_start+1
            while sub_idx < num_end:
                l_split, sub_idx, token_list = token_add_color_RGB(l_split, sub_idx, token_list)
            next_idx = num_end + 1
        elif l_split[idx+1] == '[':
            num_start = idx + 1
            num_end = find_matching_brace(l_split, num_start, brace=['[', ']'])
            den_start = num_end + 1
            den_end = find_matching_brace(l_split, den_start)
            color_token = "\\mathcolor[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
            l_split = l_split[:idx] + [color_token+token] + l_split[idx+1: den_end+1] + ["}"] + l_split[den_end+1:]
            token_list.append(token)
            sub_idx = num_start + 1
            while sub_idx < num_end:
                l_split, sub_idx, token_list = token_add_color_RGB(l_split, sub_idx, token_list, brace_color=True)
            sub_idx = den_start + 1
            while sub_idx < den_end:
                l_split, sub_idx, token_list = token_add_color_RGB(l_split, sub_idx, token_list)
            next_idx = den_end + 1
    elif token in ["\\multicolumn", "\\multirow"]:
        # ** tokens with three {}, such as \multicolumn {} {} {}, the text in third {} need be rendered.
        first_start = idx + 1
        first_end = find_matching_brace(l_split, first_start)
        second_start = first_end + 1
        second_end = find_matching_brace(l_split, second_start)
        third_start = second_end + 1
        third_end = find_matching_brace(l_split, third_start)
        
        sub_idx = third_start+1
        while sub_idx < third_end:
            l_split, sub_idx, token_list = token_add_color_RGB(l_split, sub_idx, token_list)
        next_idx = third_end + 1
    elif token in SKIP_Tokens+TWO_Tail_Invisb_Tokens or any(re.match(pattern, token) for pattern in SKIP_PATTERNS):
        # ** tokens no need render, just skip
        # 特殊情况：[] 可能是独立方括号，也可能属于 \sqrt[]{}
        if (token == "[" and l_split[idx-1]!="\\sqrt") or (token == "]" and idx>=3 and l_split[idx-3]!="\\sqrt"):
            color_token = "\\mathcolor[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
            l_split = l_split[:idx] + [color_token + l_split[idx] + "}"] + l_split[idx+1:]
            token_list.append(token)
            next_idx = idx + 1
        else:
            next_idx = idx + 1
    else:
        # ** nomal token
        if brace_color or (idx > 1 and l_split[idx-1] == "_"):
            color_token = "\\mathcolor[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
            l_split = l_split[:idx] + ["{" + color_token + l_split[idx] + "}}"] + l_split[idx+1:]
            token_list.append(token)
            next_idx = idx + 1
        else:
            color_token = "\\mathcolor[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
            l_split = l_split[:idx] + [color_token + l_split[idx] + "}"] + l_split[idx+1:]
            token_list.append(token)
            next_idx = idx + 1
    return l_split, next_idx, token_list