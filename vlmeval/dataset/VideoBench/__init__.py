from .CGAVCounting import CGAVCounting
from .cgbench import (CGBench_MCQ_Grounding, CGBench_MCQ_Grounding_Mini,
                      CGBench_OpenEnded, CGBench_OpenEnded_Mini)
from .dream1k import DREAM
from .dsrbench import DSRBench
from .EgoExoBench.egoexobench import EgoExoBench_MCQ
from .longvideobench import LongVideoBench
from .megabench import MEGABench
from .mlvu import MLVU, MLVU_MCQ, MLVU_OpenEnded
from .mmbench_video import MMBenchVideo
from .mmsi_video import MMSIVideoBench
from .mmsivideobench_easi import MMSIVideoBench_EASI
from .moviechat1k import MovieChat1k
from .mvbench import MVBench, MVBench_MP4
from .mvu_eval import MVUEval
from .omtgbench import OMTGBench
from .qbench_video import QBench_Video, QBench_Video_MCQ, QBench_Video_VQA
from .sitebench_video import SiteBenchVideo
from .stibench import STIBench
from .tamperbench import MVTamperBench
from .tempcompass import (TempCompass, TempCompass_Captioning, TempCompass_MCQ,
                          TempCompass_YorN)
from .v2pbench import V2PBench
from .vcrbench import VCRBench
from .vdc import VDC
from .video_concat_dataset import ConcatVideoDataset
from .video_holmes import Video_Holmes
from .video_mmlu import Video_MMLU_CAP, Video_MMLU_QA
from .videomme import VideoMME
from .videommmu import VideoMMMU
from .vsibench import VSIBench
from .vsibench_easi import VsiBench_EASI, VsiSuperRecall_EASI, VsiSuperCount_EASI
from .worldsense import WorldSense

__all__ = [
    "CGAVCounting",
    "CGBench_MCQ_Grounding_Mini",
    "CGBench_OpenEnded_Mini",
    "CGBench_MCQ_Grounding",
    "CGBench_OpenEnded",
    "ConcatVideoDataset",
    'DREAM',
    "DSRBench",
    "EgoExoBench_MCQ",
    "LongVideoBench",
    "MEGABench",
    "MLVU",
    "MLVU_MCQ",
    "MLVU_OpenEnded",
    "MMBenchVideo",
    'MMSIVideoBench',
    'MMSIVideoBench_EASI',
    "MovieChat1k",
    "MVBench",
    "MVBench_MP4",
    "MVTamperBench",
    "MVUEval",
    "OMTGBench",
    "QBench_Video",
    "QBench_Video_MCQ",
    "QBench_Video_VQA",
    'SiteBenchVideo',
    "STIBench",
    "TempCompass",
    "TempCompass_Captioning",
    "TempCompass_MCQ",
    "TempCompass_YorN",
    'V2PBench',
    "VCRBench",
    "VDC",
    "VideoMME",
    "VideoMMMU",
    "Video_Holmes",
    "Video_MMLU_CAP",
    "Video_MMLU_QA",
    "VSIBench",
    'VsiBench_EASI',
    'VsiSuperRecall_EASI',
    'VsiSuperCount_EASI',
    "WorldSense",
]
