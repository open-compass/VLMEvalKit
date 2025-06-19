from vlmeval.dataset import *
from functools import partial

vcrbench_dataset = {
    'VCRBench_8frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=8, pack=False),
    'VCRBench_16frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=16, pack=False),
    'VCRBench_32frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=32, pack=False),
    'VCRBench_64frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=64, pack=False),
    'VCRBench_1fps_nopack': partial(VCRBench, dataset='VCR-Bench', fps=1.0, pack=False)
}

mmbench_video_dataset = {
    'MMBench_Video_8frame_nopack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=8, pack=False),
    'MMBench_Video_8frame_pack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=8, pack=True),
    'MMBench_Video_16frame_nopack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=16, pack=False),
    'MMBench_Video_64frame_nopack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=64, pack=False),
    'MMBench_Video_1fps_nopack': partial(MMBenchVideo, dataset='MMBench-Video', fps=1.0, pack=False),
    'MMBench_Video_1fps_pack': partial(MMBenchVideo, dataset='MMBench-Video', fps=1.0, pack=True)
}

mvbench_dataset = {
    'MVBench_8frame': partial(MVBench, dataset='MVBench', nframe=8),
    'MVBench_64frame': partial(MVBench, dataset='MVBench', nframe=64),
    # MVBench not support fps, but MVBench_MP4 does
    'MVBench_MP4_8frame': partial(MVBench_MP4, dataset='MVBench_MP4', nframe=8),
    'MVBench_MP4_1fps': partial(MVBench_MP4, dataset='MVBench_MP4', fps=1.0),
}

tamperbench_dataset = {
    'MVTamperBench_8frame': partial(MVTamperBench, dataset='MVTamperBench', nframe=8),
    'MVTamperBenchStart_8frame': partial(MVTamperBench, dataset='MVTamperBenchStart', nframe=8),
    'MVTamperBenchEnd_8frame': partial(MVTamperBench, dataset='MVTamperBenchEnd', nframe=8),
}

videomme_dataset = {
    'Video-MME_8frame': partial(VideoMME, dataset='Video-MME', nframe=8),
    'Video-MME_64frame': partial(VideoMME, dataset='Video-MME', nframe=64),
    'Video-MME_8frame_subs': partial(VideoMME, dataset='Video-MME', nframe=8, use_subtitle=True),
    'Video-MME_1fps': partial(VideoMME, dataset='Video-MME', fps=1.0),
    'Video-MME_0.5fps': partial(VideoMME, dataset='Video-MME', fps=0.5),
    'Video-MME_0.5fps_subs': partial(VideoMME, dataset='Video-MME', fps=0.5, use_subtitle=True),
}

longvideobench_dataset = {
    'LongVideoBench_8frame': partial(LongVideoBench, dataset='LongVideoBench', nframe=8),
    'LongVideoBench_8frame_subs': partial(LongVideoBench, dataset='LongVideoBench', nframe=8, use_subtitle=True),
    'LongVideoBench_64frame': partial(LongVideoBench, dataset='LongVideoBench', nframe=64),
    'LongVideoBench_1fps': partial(LongVideoBench, dataset='LongVideoBench', fps=1.0),
    'LongVideoBench_0.5fps': partial(LongVideoBench, dataset='LongVideoBench', fps=0.5),
    'LongVideoBench_0.5fps_subs': partial(LongVideoBench, dataset='LongVideoBench', fps=0.5, use_subtitle=True)
}

mlvu_dataset = {
    'MLVU_8frame': partial(MLVU, dataset='MLVU', nframe=8),
    'MLVU_64frame': partial(MLVU, dataset='MLVU', nframe=64),
    'MLVU_1fps': partial(MLVU, dataset='MLVU', fps=1.0)
}

tempcompass_dataset = {
    'TempCompass_8frame': partial(TempCompass, dataset='TempCompass', nframe=8),
    'TempCompass_64frame': partial(TempCompass, dataset='TempCompass', nframe=64),
    'TempCompass_1fps': partial(TempCompass, dataset='TempCompass', fps=1.0),
    'TempCompass_0.5fps': partial(TempCompass, dataset='TempCompass', fps=0.5)
}

# In order to reproduce the experimental results in CGbench paper,
# use_subtitle, use_subtitle_time and use_frame_time need to be set to True.
# When measuring clue-related results, if the number of frames used is greater
# than 32, the frame capture limit will be set to 32.
# We implement the metrics long_acc, clue_acc, miou, CRR, acc@iou and rec@iou
# in the CGBench_MCQ_Grounding_Mini and CGBench_MCQ_Grounding datasets;
# the metric open-ended is implemented in the CGBench_OpenEnded_Mini and CGBench_OpenEnded datasets.
cgbench_dataset = {
    'CGBench_MCQ_Grounding_Mini_8frame_subs_subt': partial(
        CGBench_MCQ_Grounding_Mini,
        dataset='CG-Bench_MCQ_Grounding_Mini',
        nframe=8,
        use_subtitle=True,
        use_subtitle_time=True
    ),
    'CGBench_OpenEnded_Mini_8frame_subs_subt_ft': partial(
        CGBench_OpenEnded_Mini,
        dataset='CG-Bench_OpenEnded_Mini',
        nframe=8,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    ),
    'CGBench_MCQ_Grounding_32frame_subs': partial(
        CGBench_MCQ_Grounding,
        dataset='CG-Bench_MCQ_Grounding',
        nframe=32,
        use_subtitle=True
    ),
    'CGBench_OpenEnded_8frame': partial(
        CGBench_OpenEnded,
        dataset='CG-Bench_OpenEnded',
        nframe=8
    ),
    'CGBench_MCQ_Grounding_16frame_subs_subt_ft': partial(
        CGBench_MCQ_Grounding,
        dataset='CG-Bench_MCQ_Grounding',
        nframe=16,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    ),
    'CGBench_OpenEnded_16frame_subs_subt_ft': partial(
        CGBench_OpenEnded,
        dataset='CG-Bench_OpenEnded',
        nframe=16,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    )
}


megabench_dataset = {
    'MEGABench_core_16frame': partial(MEGABench, dataset='MEGABench', nframe=16, subset_name="core"),
    'MEGABench_open_16frame': partial(MEGABench, dataset='MEGABench', nframe=16, subset_name="open"),
    'MEGABench_core_64frame': partial(MEGABench, dataset='MEGABench', nframe=64, subset_name="core"),
    'MEGABench_open_64frame': partial(MEGABench, dataset='MEGABench', nframe=64, subset_name="open")
}

moviechat1k_dataset = {
    'moviechat1k_breakpoint_8frame': partial(MovieChat1k, dataset='MovieChat1k', subset='breakpoint', nframe=8),
    'moviechat1k_global_14frame': partial(MovieChat1k, dataset='MovieChat1k', subset='global', nframe=14),
    'moviechat1k_global_8frame_limit0.01': partial(
        MovieChat1k, dataset='MovieChat1k', subset='global', nframe=8, limit=0.01
    )
}

vdc_dataset = {
    'VDC_8frame': partial(VDC, dataset='VDC', nframe=8),
    'VDC_1fps': partial(VDC, dataset='VDC', fps=1.0),
}

worldsense_dataset = {
    'WorldSense_8frame': partial(WorldSense, dataset='WorldSense', nframe=8),
    'WorldSense_8frame_subs': partial(WorldSense, dataset='WorldSense', nframe=8, use_subtitle=True),
    'WorldSense_8frame_audio': partial(WorldSense, dataset='WorldSense', nframe=8, use_audio=True),
    'WorldSense_32frame': partial(WorldSense, dataset='WorldSense', nframe=32),
    'WorldSense_32frame_subs': partial(WorldSense, dataset='WorldSense', nframe=32, use_subtitle=True),
    'WorldSense_32frame_audio': partial(WorldSense, dataset='WorldSense', nframe=32, use_audio=True),
    'WorldSense_1fps': partial(WorldSense, dataset='WorldSense', fps=1.0),
    'WorldSense_1fps_subs': partial(WorldSense, dataset='WorldSense', fps=1.0, use_subtitle=True),
    'WorldSense_1fps_audio': partial(WorldSense, dataset='WorldSense', fps=1.0, use_audio=True),
    'WorldSense_0.5fps': partial(WorldSense, dataset='WorldSense', fps=0.5),
    'WorldSense_0.5fps_subs': partial(WorldSense, dataset='WorldSense', fps=0.5, use_subtitle=True),
    'WorldSense_0.5fps_audio': partial(WorldSense, dataset='WorldSense', fps=0.5, use_audio=True)
}

qbench_video_dataset = {
    'QBench_Video_8frame': partial(QBench_Video, dataset='QBench_Video', nframe=8),
    'QBench_Video_16frame': partial(QBench_Video, dataset='QBench_Video', nframe=16),
}

video_mmlu_dataset = {
    'VideoMMLU_CAP_16frame': partial(VideoMMLU_CAP, dataset='Video_MMLU_CAP', nframe=16),
    'VideoMMLU_CAP_64frame': partial(VideoMMLU_CAP, dataset='Video_MMLU_CAP', nframe=64),
    'VideoMMLU_QA_16frame': partial(VideoMMLU_QA, dataset='Video_MMLU_QA', nframe=16),
    'VideoMMLU_QA_64frame': partial(VideoMMLU_QA, dataset='Video_MMLU_QA', nframe=64),
}

video_holmes_dataset = {
    'Video_Holmes_32frame': partial(Video_Holmes, dataset='Video_Holmes', nframe=32),
    'Video_Holmes_64frame': partial(Video_Holmes, dataset='Video_Holmes', nframe=64),
}

supported_video_datasets = {}

dataset_groups = [
    mmbench_video_dataset, mvbench_dataset, videomme_dataset, longvideobench_dataset,
    mlvu_dataset, tempcompass_dataset, cgbench_dataset, worldsense_dataset, tamperbench_dataset,
    megabench_dataset, qbench_video_dataset, moviechat1k_dataset, vdc_dataset, video_holmes_dataset, vcrbench_dataset

]

for grp in dataset_groups:
    supported_video_datasets.update(grp)
