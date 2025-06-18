# flake8: noqa
from typing import List, Tuple
# from dotenv import load_dotenv
# load_dotenv()

import os
from eval_configs.global_config import run_script_safe



class TextEvaluator:

    def __init__(self, use_position=False, use_axs=True) -> None:
        self.metrics = {
            "precision": 0,
            "recall": 0,
            "f1": 0
        }
        self.use_position = use_position
        self.use_axs = use_axs

    def __call__(self, generation_code_file, golden_code_file):
        generation_texts = self._log_texts(generation_code_file)
        golden_texts = self._log_texts(golden_code_file)

        self._calculate_metrics(generation_texts, golden_texts)

        # [TAG] What is this for?
        # print(f"os.getcwd(): {os.getcwd()}")
        # breakpoint()
        # redunant_file = os.environ["PROJECT_PATH"] + "/" + os.path.basename(golden_code_file).replace(".py", ".pdf")
        # os.remove(redunant_file)
        # print(self.metrics)

    def _log_texts(self, code_file):
        """
        Get text objects of the code
        """

        with open(code_file, 'r') as f:
            lines = f.readlines()
        code = ''.join(lines)

        prefix = self._get_prefix()
        output_file = code_file.replace(".py", "_log_texts.txt")
        suffix = self._get_suffix(output_file)
        code = prefix + code + suffix

        if not self.use_axs:
            # find plt.savefig and append code before it
            savefig_idx = code.find("plt.savefig")
            ax_ticks_deletion_code = self._get_ax_ticks_deletion_code()
            code = code[:savefig_idx] + \
                ax_ticks_deletion_code + code[savefig_idx:]

        code_log_texts_file = code_file.replace(".py", "_log_texts.py")
        with open(code_log_texts_file, 'w') as f:
            f.write(code)

        # os.system(f"python3 {code_log_texts_file}")
        success = run_script_safe(code_log_texts_file)
        if not success:
            print("Skip downstream logic due to previous failure.")
            # optionally return default result or continue

        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                texts = f.read()
                texts = eval(texts)
            os.remove(output_file)
        else:
            texts = []
        os.remove(code_log_texts_file)

        # pdf_file = re.findall(r"plt\.savefig\('(.*)'\)", code)
        # if len(pdf_file) != 0:
        # pdf_file = pdf_file[0]
        # if os.path.basename(pdf_file) == pdf_file:
        # os.remove(pdf_file)

        return texts

    def _calculate_metrics(
            self,
            generation_texts: List[Tuple],
            golden_texts: List[Tuple]):
        """
        Calculate the metrics

        Args:
            - generation_texts: List of tuples of texts, [(x, y, x_rel, y_rel, text), ...]
            - golden_texts: List of tuples of texts, [(x, y, x_rel, y_rel, text), ...]
        """
        if len(generation_texts) == 0 or len(golden_texts) == 0:
            self.metrics["precision"] = 0
            self.metrics["recall"] = 0
            self.metrics["f1"] = 0
            return

        len_generation = len(generation_texts)
        len_golden = len(golden_texts)

        if not self.use_position:
            generation_texts = [t[-1] for t in generation_texts]
            golden_texts = [t[-1] for t in golden_texts]

            n_correct = 0
            for t in golden_texts:
                if t in generation_texts:
                    n_correct += 1
                    generation_texts.remove(t)

        else:
            generation_texts = [t[2:] for t in generation_texts]
            golden_texts = [t[2:] for t in golden_texts]

            n_correct = 0
            for t1 in golden_texts:
                for t2 in generation_texts:
                    # text must be equal, but x_rel and y_rel can be in a range
                    if t1[-1] == t2[-1] and abs(t1[0] - t2[0]
                                                ) <= 10 and abs(t1[1] - t2[1]) <= 10:
                        # print("matched:", t2)
                        n_correct += 1
                        generation_texts.remove(t2)
                        break

        self.metrics["precision"] = n_correct / len_generation
        self.metrics["recall"] = n_correct / len_golden
        if self.metrics["precision"] + self.metrics["recall"] == 0:
            self.metrics["f1"] = 0
        else:
            self.metrics["f1"] = 2 * self.metrics["precision"] * \
                self.metrics["recall"] / (self.metrics["precision"] + self.metrics["recall"])

        return

    def _get_prefix(self):
        sys_to_add = os.environ["VLMEVAL_CHARTMIMIC_UTILS_PATH"]
        # assert sys_to_add not empty
        assert sys_to_add != "", "VLMEVAL_CHARTMIMIC_UTILS_PATH is not set"
        return f"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
if "{sys_to_add}" not in sys.path:
    sys.path.insert(0, "{sys_to_add}")

import eval_configs.global_config as global_config
global_config.reset_texts()
from matplotlib.backends.backend_pdf import RendererPdf

drawed_texts = []

def log_function(func):
    def wrapper(*args, **kwargs):
        global drawed_texts

        object = args[0]
        x = args[2]
        y = args[3]
        x_rel = ( x / object.width / 72 ) * 100
        y_rel = ( y / object.height / 72 ) * 100
        s = args[4]

        drawed_texts.append( (x, y, x_rel, y_rel, s) )
        return func(*args, **kwargs)

    return wrapper

RendererPdf.draw_text = log_function(RendererPdf.draw_text)
"""

    def _get_suffix(self, output_file):
        return f"""
# print("drawed_texts", drawed_texts)
with open('{output_file}', 'w') as f:
    f.write(str(drawed_texts))
"""

    def _get_ax_ticks_deletion_code(self):
        return """
all_axes = plt.gcf().get_axes()
for ax in all_axes:
    ax.set_xticks([])
    ax.set_yticks([])
"""


if __name__ == "__main__":
    # sys.path.insert(0, '/home/yc21/project/Princess-s-CHI')

    evaluator = TextEvaluator(use_axs=False)
    # evaluator = TextEvaluator()

    generation_code_file = "/home/yc21/project/Princess-s-CHI/dataset/line/line_7.py"
    golden_code_file = "/home/yc21/project/Princess-s-CHI/results/chart2code_gpt_DirectAgent_results/direct/line_7.py"

    evaluator(generation_code_file, golden_code_file)
