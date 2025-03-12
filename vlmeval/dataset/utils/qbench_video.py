from ...smp import *
from .multiple_choice import extract_answer_from_item
from PIL import Image, ImageOps
import numpy as np

FAIL_MSG = 'Failed to obtain answer via API.'
def check_ans_mcq(pred, ans, correct_choice, correct_answer):
    flag = False

    if correct_choice == pred or correct_choice+"." in pred or correct_answer == pred:
        flag = True
    elif correct_choice in pred.split("\n"):
        flag = True

    return flag