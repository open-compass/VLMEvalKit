
import warnings
import pandas as pd
import re
import numpy as np
from abc import abstractmethod
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from ..smp import *
from .text_base import TextBaseDataset

data_path = "~/LMUData"

non_numeric_props_options = {
    'Direct_or_indirect': ['Indirect', 'Direct'],
    'Direct_or_indirect_HSE': ['Indirect', 'Direct'],
    'SOC': [True, False],
    'is_gap_direct': [True, False],
    'is_stable': [True, False],
}

chem_pattern = re.compile(r'\b(?:[A-Z][a-z]?\d*|\([^\)]+\)\d*)+\b')

def remove_chemical_expressions(text: str):
    return chem_pattern.sub('', text)


def extract_values(sentence: str):
    sentence_clean = remove_chemical_expressions(sentence).strip().lower()
    result = None

    if re.search(r'\b[+-]?inf\b', sentence_clean):
        return float('-inf') if '-inf' in sentence_clean else float('inf')

    match_1 = re.search(r'(\d+(\.\d+)?)\s*x\s*10\s*\^*\s*(-?\d+)', sentence_clean)
    match_2 = re.search(r'(\d+(\.\d+)?)\s*×\s*10\s*\^*\s*(-?\d+)', sentence_clean)
    match_3 = re.search(r'(\d+(\.\d+)?[eE][+-]?\d+)', sentence_clean)

    try:
        if match_1:
            value, _, exp = match_1.groups()
            result = float(value) * 10 ** int(exp)
        elif match_2:
            value, _, exp = match_2.groups()
            result = float(value) * 10 ** int(exp)
        elif match_3:
            result = float(match_3.group())
        elif "10^" in sentence_clean:
            match = re.search(r'(?<=\^)[+-]?\d+', sentence_clean)
            if match:
                result = 10 ** int(match.group())
        else:
            range_match = re.findall(r'(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)', sentence_clean)
            if range_match:
                nums = [float(x) for x in range_match[0]]
                result = np.mean(nums)
            else:
                numbers = re.findall(r'-?\d+\.?\d*', sentence_clean)
                if numbers:
                    result = float(numbers[0]) if '.' in numbers[0] else int(numbers[0])
    except:
        result = None

    if result is not None and result not in [float('inf'), float('-inf')] and abs(result) >= 1e5:
        result = None
    return result


def remove_think_tags(text: str) -> str:
    if "<think>" not in text:
        return text
    if "</think>" not in text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def extract_strict_or_loose_value(text: str, property: str):

    if not isinstance(text, str):
        return ""
    text = text.strip()
    if text == "":
        return ""

    pattern = rf"\{{\s*{re.escape(property)}\s*:\s*(.*?)\s*\}}"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        raw_value = match.group(1).strip()
        if property in non_numeric_props_options:
            for opt in non_numeric_props_options[property]:
                if isinstance(opt, bool):
                    if raw_value.lower() == str(opt).lower():
                        return str(opt)
                elif raw_value.lower() == opt.lower():
                    return opt
            return ""

        try:
            return float(raw_value)
        except ValueError:
            pass

    val = extract_values(text)
    if val is not None:
        return val
    return ""

LLM4Mat_sub_tasks = {
    'MP_FEPA': {'property': 'formation_energy_per_atom', 'file_name': 'MP_FEPA.tsv'},
    'MP_Bandgap': {'property': 'band_gap', 'file_name': 'MP_Bandgap.tsv'},
    'MP_EPA': {'property': 'GGA-PBE-based_energy_per_atom', 'file_name': 'MP_EPA.tsv'},
    'MP_Ehull': {'property': 'energy_above_hull', 'file_name': 'MP_Ehull.tsv'},
    'MP_Efermi': {'property': 'efermi', 'file_name': 'MP_Efermi.tsv'},
    'MP_Density': {'property': 'density', 'file_name': 'MP_Density.tsv'},
    'MP_DensityAtomic': {'property': 'density_atomic', 'file_name': 'MP_DensityAtomic.tsv'},
    'MP_Volume': {'property': 'volume', 'file_name': 'MP_Volume.tsv'},
    'MP_IsStable': {'property': 'is_stable', 'file_name': 'MP_IsStable.tsv'},
    'MP_IsGapDirect': {'property': 'is_gap_direct', 'file_name': 'MP_IsGapDirect.tsv'},

    'JARVISDFT_FEPA': {'property': 'formation_energy_peratom', 'file_name': 'JARVISDFT_FEPA.tsv'},
    'JARVISDFT_Bandgap_OPT': {'property': 'optb88vdw_bandgap', 'file_name': 'JARVISDFT_Bandgap_OPT.tsv'},
    'JARVISDFT_TotEn': {'property': 'optb88vdw_total_energy', 'file_name': 'JARVISDFT_TotEn.tsv'},
    'JARVISDFT_Ehull': {'property': 'ehull', 'file_name': 'JARVISDFT_Ehull.tsv'},
    'JARVISDFT_Bandgap_MBJ': {'property': 'mbj_bandgap', 'file_name': 'JARVISDFT_Bandgap_MBJ.tsv'},
    'JARVISDFT_Kv': {'property': 'bulk_modulus_kv', 'file_name': 'JARVISDFT_Kv.tsv'},
    'JARVISDFT_Gv': {'property': 'shear_modulus_gv', 'file_name': 'JARVISDFT_Gv.tsv'},
    'JARVISDFT_SLME': {'property': 'slme', 'file_name': 'JARVISDFT_SLME.tsv'},
    'JARVISDFT_Spillage': {'property': 'spillage', 'file_name': 'JARVISDFT_Spillage.tsv'},
    'JARVISDFT_Epsx_OPT': {'property': 'mepsx', 'file_name': 'JARVISDFT_Epsx_OPT.tsv'},
    'JARVISDFT_Dielectric_DFPT': {'property': 'dfpt_piezo_max_dielectric', 'file_name': 'JARVISDFT_Dielectric_DFPT.tsv'},
    'JARVISDFT_Max_Piezo_dij': {'property': 'dfpt_piezo_max_dij', 'file_name': 'JARVISDFT_Max_Piezo_dij.tsv'},
    'JARVISDFT_Max_Piezo_eij': {'property': 'dfpt_piezo_max_eij', 'file_name': 'JARVISDFT_Max_Piezo_eij.tsv'},
    'JARVISDFT_MaxEFG': {'property': 'max_efg', 'file_name': 'JARVISDFT_MaxEFG.tsv'},
    'JARVISDFT_ExfEn': {'property': 'exfoliation_energy', 'file_name': 'JARVISDFT_ExfEn.tsv'},
    'JARVISDFT_AvgMe': {'property': 'avg_elec_mass', 'file_name': 'JARVISDFT_AvgMe.tsv'},
    'JARVISDFT_nSeebeck': {'property': 'n-Seebeck', 'file_name': 'JARVISDFT_nSeebeck.tsv'},
    'JARVISDFT_nPF': {'property': 'n-powerfact', 'file_name': 'JARVISDFT_nPF.tsv'},
    'JARVISDFT_pSeebeck': {'property': 'p-Seebeck', 'file_name': 'JARVISDFT_pSeebeck.tsv'},
    'JARVISDFT_pPF': {'property': 'p-powerfact', 'file_name': 'JARVISDFT_pPF.tsv'},

    'SNUMAT_Bandgap_GGA': {'property': 'Band_gap_GGA', 'file_name': 'SNUMAT_Bandgap_GGA.tsv'},
    'SNUMAT_Bandgap_HSE': {'property': 'Band_gap_HSE', 'file_name': 'SNUMAT_Bandgap_HSE.tsv'},
    'SNUMAT_Bandgap_GGA_Optical': {'property': 'Band_gap_GGA_optical', 'file_name': 'SNUMAT_Bandgap_GGA_Optical.tsv'},
    'SNUMAT_Bandgap_HSE_Optical': {'property': 'Band_gap_HSE_optical', 'file_name': 'SNUMAT_Bandgap_HSE_Optical.tsv'},
    'SNUMAT_IsDirect': {'property': 'Direct_or_indirect', 'file_name': 'SNUMAT_IsDirect.tsv'},
    'SNUMAT_IsDirect_HSE': {'property': 'Direct_or_indirect_HSE', 'file_name': 'SNUMAT_IsDirect_HSE.tsv'},
    'SNUMAT_SOC': {'property': 'SOC', 'file_name': 'SNUMAT_SOC.tsv'},

    'GNoME_FEPA': {'property': 'Formation_Energy_Per_Atom', 'file_name': 'GNoME_FEPA.tsv'},
    'GNoME_DEPA': {'property': 'Decomposition_Energy_Per_Atom', 'file_name': 'GNoME_DEPA.tsv'},
    'GNoME_Bandgap': {'property': 'Bandgap', 'file_name': 'GNoME_Bandgap.tsv'},
    'GNoME_TotEn': {'property': 'Corrected_Energy', 'file_name': 'GNoME_TotEn.tsv'},
    'GNoME_Volume': {'property': 'Volume', 'file_name': 'GNoME_Volume.tsv'},
    'GNoME_Density': {'property': 'Density', 'file_name': 'GNoME_Density.tsv'},

    'hMOF_MaxCO2': {'property': 'max_co2_adsp', 'file_name': 'hMOF_MaxCO2.tsv'},
    'hMOF_MinCO2': {'property': 'min_co2_adsp', 'file_name': 'hMOF_MinCO2.tsv'},
    'hMOF_LCD': {'property': 'lcd', 'file_name': 'hMOF_LCD.tsv'},
    'hMOF_PLD': {'property': 'pld', 'file_name': 'hMOF_PLD.tsv'},
    'hMOF_VoidFraction': {'property': 'void_fraction', 'file_name': 'hMOF_VoidFraction.tsv'},
    'hMOF_SA_m2g': {'property': 'surface_area_m2g', 'file_name': 'hMOF_SA_m2g.tsv'},
    'hMOF_SA_m2cm3': {'property': 'surface_area_m2cm3', 'file_name': 'hMOF_SA_m2cm3.tsv'},

    'Cantor_HEA_FEPA': {'property': 'Ef_per_atom', 'file_name': 'Cantor_HEA_FEPA.tsv'},
    'Cantor_HEA_EPA': {'property': 'e_per_atom', 'file_name': 'Cantor_HEA_EPA.tsv'},
    'Cantor_HEA_Ehull': {'property': 'e_above_hull', 'file_name': 'Cantor_HEA_Ehull.tsv'},
    'Cantor_HEA_VPA': {'property': 'volume_per_atom', 'file_name': 'Cantor_HEA_VPA.tsv'},

    'QMOF_TotEn': {'property': 'energy_total', 'file_name': 'QMOF_TotEn.tsv'},
    'QMOF_Bandgap': {'property': 'bandgap', 'file_name': 'QMOF_Bandgap.tsv'},
    'QMOF_LCD': {'property': 'lcd', 'file_name': 'QMOF_LCD.tsv'},
    'QMOF_PLD': {'property': 'pld', 'file_name': 'QMOF_PLD.tsv'},

    'JARVISQETB_EPA': {'property': 'TB-based_energy_per_atom', 'file_name': 'JARVISQETB_EPA.tsv'},
    'JARVISQETB_IndirBandgap': {'property': 'indir_gap', 'file_name': 'JARVISQETB_IndirBandgap.tsv'},
    'JARVISQETB_FEPA': {'property': 'f_enp', 'file_name': 'JARVISQETB_FEPA.tsv'},
    'JARVISQETB_TotEn': {'property': 'final_energy', 'file_name': 'JARVISQETB_TotEn.tsv'},

    'OQMD_Bandgap': {'property': 'bandgap', 'file_name': 'OQMD_Bandgap.tsv'},
    'OQMD_FEPA': {'property': 'e_form', 'file_name': 'OQMD_FEPA.tsv'},

    'OMDB_Bandgap': {'property': 'bandgap', 'file_name': 'OMDB_Bandgap.tsv'}
}


class LLM4Mat(TextBaseDataset):
    TYPE = 'TEXT'
    DATASET_URL = {task: f"{data_path}/{info['file_name']}" for task, info in LLM4Mat_sub_tasks.items()}
    DATASET_MD5 = {task: "" for task, _ in LLM4Mat_sub_tasks.items()}

    @classmethod
    def postprocess(cls, text: str, property: str):
        if text is None or not isinstance(text, str):
            return ""
        text = remove_think_tags(text.strip())
        if text == "":
            return ""
        return extract_strict_or_loose_value(text, property)

    @classmethod
    def score(cls, predictions, references):
        valid_refs = [r for r in references if r not in [None, "Null"]]
        if len(valid_refs) == 0:
            return {}

        is_regression = isinstance(valid_refs[0], (int, float)) and not isinstance(valid_refs[0], bool)

        if is_regression:
            y_true, y_pred = [], []
            total = len(references)
            for t, p in zip(references, predictions):
                try:
                    t_val = float(t)
                    p_val = float(p)
                    if not (np.isfinite(t_val) and np.isfinite(p_val)):
                        continue
                    y_true.append(t_val)
                    y_pred.append(p_val)
                except:
                    continue

            if len(y_true) == 0:
                return {"MAE": None, "RMSE": None, "MAD": None, "MAD/MAE": None}

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mean_value = np.mean(y_true)
            mad = mean_absolute_error(y_true, [mean_value] * len(y_true))
            return {
                "total": total,
                "filtered": len(y_true),
                "MAE": mae,
                "RMSE": rmse,
                "MAD": mad,
                "MAD/MAE": mad / mae if mae != 0 else None,
            }

        y_true, y_pred = [], []
        auc = None
        try:
            for t, p in zip(references, predictions):
                if t in ["Null"]:
                    continue
                if t in ["Direct", "True", True]:
                    y_true.append(1)
                elif t in ["Indirect", "False", False]:
                    y_true.append(0)
                else:
                    continue

                if p in ["Direct", "True", True]:
                    y_pred.append(1)
                elif p in ["Indirect", "False", False]:
                    y_pred.append(0)
                else:
                    y_true.pop()
                    continue
            auc = roc_auc_score(y_true, y_pred)
        except:
            pass
        return {"AUC": auc}

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        data = load(eval_file)
        data = data[~pd.isna(data["prediction"])]
        predictions = data["prediction"].tolist()
        references = data["answer"].tolist()

        task_name = None
        for k in LLM4Mat_sub_tasks.keys():
            if k in eval_file:
                task_name = k
                break
        if task_name is None:
            raise ValueError(f"Cannot infer task_name from eval_file: {eval_file}")

        property_name = LLM4Mat_sub_tasks[task_name]["property"]

        predictions_pp = [cls.postprocess(p, property_name) for p in predictions]
        references_pp = [cls.postprocess(r, property_name) for r in references]

        metrics = cls.score(predictions_pp, references_pp)
        return metrics

