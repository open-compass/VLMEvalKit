import warnings

from .image_base import img_root_map, ImageBaseDataset
from .image_caption import ImageCaptionDataset
from .image_yorn import ImageYORNDataset
from .image_mcq import (
    ImageMCQDataset, MMMUDataset, CustomMCQDataset, MUIRDataset, GMAIMMBenchDataset, MMERealWorld, HRBenchDataset,
    NaturalBenchDataset, WeMath, MMMUProDataset, VMCBenchDataset, MedXpertQA_MM_test, LEGO, VisuLogic, CVBench, TDBench,
    MicroBench, OmniMedVQA, MSEarthMCQ, VLMBlind, SCAM, _3DSRBench, AffordanceDataset, OmniEarthMCQBench, XLRSBench
)
from .image_mt import MMDUDataset
from .image_vqa import (
    ImageVQADataset, MathVision, OCRBench, MathVista, LLaVABench, VGRPBench, MMVet, MTVQADataset, TableVQABench,
    CustomVQADataset, CRPE, MathVerse, OlympiadBench, QSpatial, VizWiz, MMNIAH, LogicVista, MME_CoT,
    MMSci_Captioning, Physics_yale, TDBenchGrounding, WildDocBenchmark, OCR_Reasoning, PhyX, CountBenchQA,
    ZEROBench, Omni3DBench, TallyQA, MMEReasoning, MMVMBench
)

from .image_ccocr import CCOCRDataset
from .image_shortqa import ImageShortQADataset, PathVQA_VAL, PathVQA_TEST
from .text_mcq import CustomTextMCQDataset, TextMCQDataset

from .vcr import VCRDataset
from .mmlongbench import MMLongBench
from .dude import DUDE
from .slidevqa import SlideVQA
from .vl_rewardbench import VLRewardBench
from .vlm2bench import VLM2Bench
from .spatial457 import Spatial457
from .charxiv import CharXiv

from .mmbench_video import MMBenchVideo
from .videomme import VideoMME
from .video_holmes import Video_Holmes
from .mvbench import MVBench, MVBench_MP4
from .tamperbench import MVTamperBench
from .miabench import MIABench
from .mlvu import MLVU, MLVU_MCQ, MLVU_OpenEnded
from .tempcompass import TempCompass, TempCompass_Captioning, TempCompass_MCQ, TempCompass_YorN
from .longvideobench import LongVideoBench
from .video_concat_dataset import ConcatVideoDataset
from .mmgenbench import MMGenBench
from .cgbench import CGBench_MCQ_Grounding_Mini, CGBench_OpenEnded_Mini, CGBench_MCQ_Grounding, CGBench_OpenEnded
from .megabench import MEGABench
from .moviechat1k import MovieChat1k
from .video_mmlu import VideoMMLU_CAP, VideoMMLU_QA
from .vdc import VDC
from .vcrbench import VCRBench
from .gobench import GOBenchDataset
from .sfebench import SFE


from .worldsense import WorldSense
from .qbench_video import QBench_Video, QBench_Video_MCQ, QBench_Video_VQA

from .cmmmu import CMMMU
from .emma import EMMADataset
from .wildvision import WildVision
from .mmmath import MMMath
from .dynamath import Dynamath
from .creation import CreationMMBenchDataset
from .mmalignbench import MMAlignBench
from .utils import *
from .video_dataset_config import *
from ..smp import *
from .Omnidocbench.omnidocbench import OmniDocBench
from .moat import MOAT
from .GUI.screenspot import ScreenSpot
from .GUI.screenspot_pro import ScreenSpot_Pro
from .mmifeval import MMIFEval
from .chartmimic import ChartMimic


class ConcatDataset(ImageBaseDataset):
    # This dataset takes multiple dataset names as input and aggregate them into a single dataset.
    # Each single dataset should not have a field named `SUB_DATASET`

    DATASET_SETS = {
        'MMMB': ['MMMB_ar', 'MMMB_cn', 'MMMB_en', 'MMMB_pt', 'MMMB_ru', 'MMMB_tr'],
        'MTL_MMBench_DEV': [
            'MMBench_dev_ar', 'MMBench_dev_cn', 'MMBench_dev_en',
            'MMBench_dev_pt', 'MMBench_dev_ru', 'MMBench_dev_tr'
        ],
        'ScreenSpot_Pro': [
            'ScreenSpot_Pro_Development', 'ScreenSpot_Pro_Creative', 'ScreenSpot_Pro_CAD',
            'ScreenSpot_Pro_Scientific', 'ScreenSpot_Pro_Office', 'ScreenSpot_Pro_OS'
        ],
        'ScreenSpot': ['ScreenSpot_Mobile', 'ScreenSpot_Desktop', 'ScreenSpot_Web'],
        'ScreenSpot_v2': ['ScreenSpot_v2_Mobile', 'ScreenSpot_v2_Desktop', 'ScreenSpot_v2_Web'],
    }

    def __init__(self, dataset):
        datasets = self.DATASET_SETS[dataset]
        self.dataset_map = {}
        # The name of the compliation
        self.dataset_name = dataset
        self.datasets = datasets
        for dname in datasets:
            dataset = build_dataset(dname)
            assert dataset is not None, dataset
            self.dataset_map[dname] = dataset
        TYPES = [x.TYPE for x in self.dataset_map.values()]
        MODALITIES = [x.MODALITY for x in self.dataset_map.values()]
        assert np.all([x == TYPES[0] for x in TYPES]), (datasets, TYPES)
        assert np.all([x == MODALITIES[0] for x in MODALITIES]), (datasets, MODALITIES)
        self.TYPE = TYPES[0]
        self.MODALITY = MODALITIES[0]
        data_all = []
        for dname in datasets:
            data = self.dataset_map[dname].data
            data['SUB_DATASET'] = [dname] * len(data)
            if 'image' in data:
                data_new = localize_df(data, dname, nproc=16)
                data_all.append(data_new)
            else:
                data_all.append(data)

        data = pd.concat(data_all)
        data['original_index'] = data.pop('index')
        data['index'] = np.arange(len(data))
        self.data = data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        idx = line['original_index']
        dname = line['SUB_DATASET']
        org_data = self.dataset_map[dname].data
        org_line = cp.deepcopy(org_data[org_data['index'] == idx]).iloc[0]
        return self.dataset_map[dname].build_prompt(org_line)

    def dump_image(self, line):
        # Assert all images are pre-dumped
        assert 'image' not in line
        assert 'image_path' in line
        tgt_path = toliststr(line['image_path'])
        return tgt_path

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_SETS)

    def evaluate(self, eval_file, **judge_kwargs):
        suffix = eval_file.split('.')[-1]
        # First, split the eval_file by dataset
        data_all = load(eval_file)
        for dname in self.datasets:
            tgt = eval_file.replace(self.dataset_name, dname)
            data_sub = data_all[data_all['SUB_DATASET'] == dname]
            data_sub.pop('index')
            data_sub['index'] = data_sub.pop('original_index')
            data_sub.pop('SUB_DATASET')
            dump(data_sub, tgt)
        # Then, evaluate each dataset separately
        df_all = []
        dict_all = {}
        # One of the vars will be used to aggregate results
        for dname in self.datasets:
            tgt = eval_file.replace(self.dataset_name, dname)
            res = self.dataset_map[dname].evaluate(tgt, **judge_kwargs)
            if isinstance(res, pd.DataFrame):
                res['DATASET'] = [dname] * len(res)
                df_all.append(res)
            elif isinstance(res, dict):
                res = {f'{dname}:{k}': v for k, v in res.items()}
                dict_all.update(res)
            else:
                raise NotImplementedError(f'Unknown result type {type(res)}')

        if len(df_all):
            result = pd.concat(df_all)
            score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
            dump(result, score_file)
            return result
        else:
            score_file = eval_file.replace(f'.{suffix}', '_score.json')
            dump(dict_all, score_file)
            return dict_all


# Add new supported dataset class here
IMAGE_DATASET = [
    ImageCaptionDataset, ImageYORNDataset, ImageMCQDataset, ImageVQADataset,
    MathVision, MMMUDataset, OCRBench, MathVista, LLaVABench, VGRPBench, MMVet,
    MTVQADataset, TableVQABench, MMLongBench, VCRDataset, MMDUDataset, DUDE,
    SlideVQA, MUIRDataset, CCOCRDataset, GMAIMMBenchDataset, MMERealWorld,
    HRBenchDataset, CRPE, MathVerse, NaturalBenchDataset, MIABench,
    OlympiadBench, WildVision, MMMath, QSpatial, Dynamath, MMGenBench, VizWiz,
    MMNIAH, CMMMU, VLRewardBench, WeMath, LogicVista, MMMUProDataset,
    CreationMMBenchDataset, ImageShortQADataset, MMAlignBench, OmniDocBench,
    VLM2Bench, VMCBenchDataset, EMMADataset, MME_CoT, MOAT, MedXpertQA_MM_test,
    LEGO, MMSci_Captioning, Physics_yale, ScreenSpot_Pro, ScreenSpot,
    MMIFEval, Spatial457, VisuLogic, CVBench, PathVQA_VAL,
    PathVQA_TEST, TDBench, TDBenchGrounding, MicroBench, CharXiv, OmniMedVQA,
    WildDocBenchmark, MSEarthMCQ, OCR_Reasoning, PhyX, VLMBlind, CountBenchQA,
    ZEROBench, SCAM, Omni3DBench, TallyQA, _3DSRBench, AffordanceDataset,
    MMEReasoning, GOBenchDataset, SFE, ChartMimic, MMVMBench, XLRSBench, 
    OmniEarthMCQBench
]


VIDEO_DATASET = [
    MMBenchVideo, VideoMME, MVBench, MVBench_MP4, MVTamperBench,
    LongVideoBench, WorldSense, VDC, MovieChat1k, MEGABench,
    MLVU, MLVU_MCQ, MLVU_OpenEnded,
    TempCompass, TempCompass_MCQ, TempCompass_Captioning, TempCompass_YorN,
    CGBench_MCQ_Grounding_Mini, CGBench_OpenEnded_Mini, CGBench_MCQ_Grounding, CGBench_OpenEnded,
    QBench_Video, QBench_Video_MCQ, QBench_Video_VQA,
    VideoMMLU_CAP, VideoMMLU_QA,
    Video_Holmes, VCRBench
]

TEXT_DATASET = [
    TextMCQDataset
]

CUSTOM_DATASET = [
    CustomMCQDataset, CustomVQADataset, CustomTextMCQDataset
]

DATASET_COLLECTION = [ConcatDataset, ConcatVideoDataset]

DATASET_CLASSES = IMAGE_DATASET + VIDEO_DATASET + TEXT_DATASET + CUSTOM_DATASET + DATASET_COLLECTION  # noqa: E501
SUPPORTED_DATASETS = []
for DATASET_CLS in DATASET_CLASSES:
    SUPPORTED_DATASETS.extend(DATASET_CLS.supported_datasets())


def DATASET_TYPE(dataset, *, default: str = 'MCQ') -> str:
    for cls in DATASET_CLASSES:
        if dataset in cls.supported_datasets():
            if hasattr(cls, 'TYPE'):
                return cls.TYPE
    # Have to add specific routine to handle ConcatDataset
    if dataset in ConcatDataset.DATASET_SETS:
        dataset_list = ConcatDataset.DATASET_SETS[dataset]
        TYPES = [DATASET_TYPE(dname) for dname in dataset_list]
        assert np.all([x == TYPES[0] for x in TYPES]), (dataset_list, TYPES)
        return TYPES[0]

    if 'openended' in dataset.lower():
        return 'VQA'
    warnings.warn(f'Dataset {dataset} is a custom one and not annotated as `openended`, will treat as {default}. ')  # noqa: E501
    return default


def DATASET_MODALITY(dataset, *, default: str = 'IMAGE') -> str:
    if dataset is None:
        warnings.warn(f'Dataset is not specified, will treat modality as {default}. ')
        return default
    for cls in DATASET_CLASSES:
        if dataset in cls.supported_datasets():
            if hasattr(cls, 'MODALITY'):
                return cls.MODALITY
    # Have to add specific routine to handle ConcatDataset
    if dataset in ConcatDataset.DATASET_SETS:
        dataset_list = ConcatDataset.DATASET_SETS[dataset]
        MODALITIES = [DATASET_MODALITY(dname) for dname in dataset_list]
        assert np.all([x == MODALITIES[0] for x in MODALITIES]), (dataset_list, MODALITIES)
        return MODALITIES[0]

    if 'VIDEO' in dataset.lower():
        return 'VIDEO'
    elif 'IMAGE' in dataset.lower():
        return 'IMAGE'
    warnings.warn(f'Dataset {dataset} is a custom one, will treat modality as {default}. ')
    return default


def build_dataset(dataset_name, **kwargs):
    for cls in DATASET_CLASSES:
        if dataset_name in supported_video_datasets:
            return supported_video_datasets[dataset_name](**kwargs)
        elif dataset_name in cls.supported_datasets():
            return cls(dataset=dataset_name, **kwargs)

    warnings.warn(f'Dataset {dataset_name} is not officially supported. ')
    data_file = osp.join(LMUDataRoot(), f'{dataset_name}.tsv')
    if not osp.exists(data_file):
        warnings.warn(f'Data file {data_file} does not exist. Dataset building failed. ')
        return None

    data = load(data_file)
    if 'question' not in [x.lower() for x in data.columns]:
        warnings.warn(f'Data file {data_file} does not have a `question` column. Dataset building failed. ')
        return None

    if 'A' in data and 'B' in data:
        if 'image' in data or 'image_path' in data:
            warnings.warn(f'Will assume unsupported dataset {dataset_name} as a Custom MCQ dataset. ')
            return CustomMCQDataset(dataset=dataset_name, **kwargs)
        else:
            warnings.warn(f'Will assume unsupported dataset {dataset_name} as a Custom Text MCQ dataset. ')
            return CustomTextMCQDataset(dataset=dataset_name, **kwargs)
    else:
        warnings.warn(f'Will assume unsupported dataset {dataset_name} as a Custom VQA dataset. ')
        return CustomVQADataset(dataset=dataset_name, **kwargs)


def infer_dataset_basename(dataset_name):
    basename = "_".join(dataset_name.split("_")[:-1])
    return basename


__all__ = [
    'build_dataset', 'img_root_map', 'build_judge', 'extract_answer_from_item', 'prefetch_answer', 'DEBUG_MESSAGE'
] + [cls.__name__ for cls in DATASET_CLASSES]
