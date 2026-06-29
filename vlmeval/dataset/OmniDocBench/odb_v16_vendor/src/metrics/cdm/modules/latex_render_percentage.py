import re
import os
import json
import time
import shutil
import random
import argparse
import subprocess
import shlex
import numpy as np

from tqdm import tqdm
from multiprocessing import Pool

try:
    from .texlive_env import build_tex_env, resolve_tex_binary
except ImportError:
    from texlive_env import build_tex_env, resolve_tex_binary


formular_template = r"""
\documentclass[12pt]{article}
\usepackage[landscape]{geometry}
\usepackage{geometry}
\geometry{a5paper,scale=0.98}
\pagestyle{empty}
\usepackage{booktabs}
\usepackage{amsmath}
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

def run_shell_cmd(cmd, max_time=15):
    child = subprocess.Popen(cmd, shell=True, env=build_tex_env())
    for i in range(max_time):
        if child.poll():
            return True
        if i == max_time-1:
            child.kill()
            return False
        time.sleep(1)
    return False

def render_latex(latex_code, basename, latex_dir, pdf_dir):
    latex_path = os.path.join(latex_dir, basename + ".tex")
    pdf_path = os.path.join(pdf_dir, basename + ".pdf")
    with open(latex_path, "w") as f:
        f.write(formular_template % latex_code)
    pdflatex_bin = shlex.quote(resolve_tex_binary("pdflatex"))
    cmd = (
        f"{pdflatex_bin} -interaction=nonstopmode "
        f"-output-directory={shlex.quote(pdf_dir)} -output-format=pdf {shlex.quote(latex_path)} >/dev/null"
    )
    run_shell_cmd(cmd)
    return pdf_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default='data/pred_results/test.json')
    parser.add_argument('--clean', action='store_true', default=False)
    parser.add_argument('--gt', action='store_true', default=False)
    args = parser.parse_args()
    
    if args.gt:
        output_path = os.path.join("output", 'gt.json')
        load_key = 'gt'
    else:
        load_key = 'pred'
        output_path = os.path.join("output", os.path.basename(args.input))
        
    
    temp_dir=f"render_temp_dir"
    try:
        shutil.rmtree(temp_dir)
    except:
        pass
    latex_dir = os.path.join(temp_dir, "texes")
    pdf_dir = os.path.join(temp_dir, "pdfs")
    os.makedirs(latex_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    
    with open(args.input, "r") as f:
        input_data = json.load(f)
    
    myP = Pool(200)
    for idx, item in enumerate(input_data):
        basename = f"sample_{idx}"
        myP.apply_async(render_latex, args=(item[load_key], basename, latex_dir, pdf_dir))
    myP.close()
    print("processing, may take some times.")
    myP.join()
    
    success_num = 0
    total_num = 0
    for idx, item in enumerate(input_data):
        basename = f"sample_{idx}"
        total_num += 1
        pdf_path = os.path.join(pdf_dir, basename + ".pdf")
        if os.path.exists(pdf_path):
            success_num += 1
            item['renderable'] = 1
        else:
            item['renderable'] = 0
        
    print("total num:", total_num, "render success num:", success_num)
    with open(output_path, "w") as f:
        f.write(json.dumps(input_data, indent=2))
    if args.clean:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
