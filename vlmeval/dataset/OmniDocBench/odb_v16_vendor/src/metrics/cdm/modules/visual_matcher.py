import time
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class SimpleAffineTransform:
    """
    simple affine transform, only translation and scale.
    """
    def __init__(self, translation=(0, 0), scale=1.0):
        self.translation = np.array(translation)
        self.scale = scale

    def estimate(self, src, dst):
        src_center = np.mean(src, axis=0)
        dst_center = np.mean(dst, axis=0)
        self.translation = dst_center - src_center

        src_dists = np.linalg.norm(src - src_center, axis=1)
        dst_dists = np.linalg.norm(dst - dst_center, axis=1)
        self.scale = np.mean(dst_dists) / (np.mean(src_dists) + 1e-10)

    def inverse(self):
        inverse_transform = AffineTransform(-self.translation, 1.0/self.scale)
        return inverse_transform

    def __call__(self, coords):
        return self.scale * (coords - np.mean(coords, axis=0)) + np.mean(coords, axis=0) + self.translation

    def residuals(self, src, dst):
        return np.sqrt(np.sum((self(src) - dst) ** 2, axis=1))


def norm_coords(x, left, right):
    if x < left:
        return left
    if x > right:
        return right
    return x

def norm_same_token(token):
    special_map = {
        "\\dot": ".",
        "\\Dot": ".",
        "\\cdot": ".",
        "\\cdotp": ".",
        "\\ldotp": ".",
        "\\mid": "|",
        "\\rightarrow": "\\to",
        "\\top": "T",
        "\\Tilde": "\\tilde",
        "\\prime": "'",
        "\\ast": "*",
        "\\left<": "\\langle",
        "\\right>": "\\rangle",
        "\\lbrace": "\{",
        "\\rbrace": "\}",
        "\\lbrack": "[",
        "\\rbrack": "]",
        "\\blackslash": "/",
        "\\slash": "/",
        "\\leq": "\\le",
        "\\geq": "\\ge",
        "\\neq": "\\ne",
        "\\Vert": "\\|",
        "\\lVert": "\\|",
        "\\rVert": "\\|",
        "\\vert": "|",
        "\\lvert": "|",
        "\\rvert": "|",
        "\\colon": ":",
        "\\Ddot": "\\ddot",
        "\\Bar": "\\bar",
        "\\Vec": "\\vec",
        "\\parallel": "\\|",
        "\\dag": "\\dagger",
        "\\ddag": "\\ddagger",
        "\\textlangle": "<",
        "\\textrangle": ">",
        "\\textgreater": ">",
        "\\textless": "<",
        "\\textbackslash": "n",
        "\\textunderscore": "_",
        "\\=": "_",
        "\\neg": "\\lnot",
        "\\neq": "\\not=",
    }
    if token.startswith('\\left') or token.startswith('\\right'):
        if "arrow" not in token and "<" not in token and ">" not in token and "harpoon" not in token:
            token = token.replace("\\left", "").replace("\\right", "")
    if token.startswith('\\big') or token.startswith('\\Big'):
        if "\\" in token[4:]:
            token = "\\"+token[4:].split("\\")[-1]
        else:
            token = token[-1]
    if token in special_map.keys():
        token = special_map[token]
    if token.startswith('\\wide'):
        return token.replace("wide", "")
    if token.startswith('\\var'):
        return token.replace("var", "")
    if token.startswith('\\string'):
        return token.replace("\\string", "")
    return token


class HungarianMatcher:
    def __init__(
        self, 
        cost_token: float = 1,
        cost_position: float = 0.05,
        cost_order: float = 0.15,
    ):
        self.cost_token = cost_token
        self.cost_position = cost_position
        self.cost_order = cost_order
        self.cost = {}
        
    def calculate_token_cost(self, box_gt, box_pred):
        token2id = {}
        for data in box_gt+box_pred:
            if data['token'] not in token2id:
                token2id[data['token']] = len(token2id)
        num_classes = len(token2id)
        
        token2id_norm = {}
        for data in box_gt+box_pred:
            if norm_same_token(data['token']) not in token2id_norm:
                token2id_norm[norm_same_token(data['token'])] = len(token2id_norm)
        num_classes_norm = len(token2id_norm)
        
        gt_token_array = []
        norm_gt_token_array = []    
        for data in box_gt:
            gt_token_array.append(token2id[data['token']])
            norm_gt_token_array.append(token2id_norm[norm_same_token(data['token'])])
            
        pred_token_logits = []
        norm_pred_token_logits = []
        for data in box_pred:
            logits = [0] * num_classes
            logits[token2id[data['token']]] = 1
            pred_token_logits.append(logits)
            
            logits_norm = [0] * num_classes_norm
            logits_norm[token2id_norm[norm_same_token(data['token'])]] = 1
            norm_pred_token_logits.append(logits_norm)
            
        gt_token_array = np.array(gt_token_array)
        pred_token_logits = np.array(pred_token_logits)
        
        norm_gt_token_array = np.array(norm_gt_token_array)
        norm_pred_token_logits = np.array(norm_pred_token_logits)
        
        token_cost = 1.0 - pred_token_logits[:, gt_token_array]
        norm_token_cost = 1.0 - norm_pred_token_logits[:, norm_gt_token_array]

        token_cost[np.logical_and(token_cost==1, norm_token_cost==0)] = 0.005
        return token_cost.T
        
        
    def box2array(self, box_list, size):
        W, H = size
        box_array = []
        for box in box_list:
            x_min, y_min, x_max, y_max = box['bbox']
            box_array.append([x_min/W, y_min/H, x_max/W, y_max/H])
        return np.array(box_array)
        
    def order2array(self, box_list, max_token_lens=None):
        if not max_token_lens:
            max_token_lens = len(box_list)
        order_array = []
        for idx, box in enumerate(box_list):
            order_array.append([idx / max_token_lens])
        return np.array(order_array)
    
    def calculate_l1_cost(self, gt_array, pred_array):
        scale = gt_array.shape[-1]
        l1_cost = cdist(gt_array, pred_array, 'minkowski', p=1)
        return l1_cost / scale
        
    def __call__(self, box_gt, box_pred, gt_size, pred_size):
        aa = time.time()
        gt_box_array = self.box2array(box_gt, gt_size)
        pred_box_array = self.box2array(box_pred, pred_size)

        max_token_lens = max(len(box_gt), len(box_pred))
        gt_order_array = self.order2array(box_gt, max_token_lens)
        pred_order_array = self.order2array(box_pred, max_token_lens)

        token_cost = self.calculate_token_cost(box_gt, box_pred)
        position_cost = self.calculate_l1_cost(gt_box_array, pred_box_array)
        order_cost = self.calculate_l1_cost(gt_order_array, pred_order_array)

        self.cost["token"] = token_cost
        self.cost["position"] = position_cost
        self.cost["order"] = order_cost
        
        cost = self.cost_token * token_cost + self.cost_position * position_cost + self.cost_order * order_cost
        cost[np.isnan(cost) | np.isinf(cost)] = 100
        indexes = linear_sum_assignment(cost)
        matched_idxes = []
        for a, b in zip(*indexes):
            matched_idxes.append((a, b))
        
        return matched_idxes