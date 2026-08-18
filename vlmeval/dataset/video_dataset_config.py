from vlmeval.dataset import *
from vlmeval.smp.dataset_alias import DatasetSpec


def _video_spec(cls, **kwargs):
    dataset = kwargs.get('dataset')
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError(
            f'Predefined video dataset for {cls.__name__} must set a non-empty dataset'
        )
    return DatasetSpec(
        dataset_alias_name='',
        dataset_name=dataset,
        dataset_class_name=cls.__name__,
        build_config={'class': cls.__name__, **kwargs},
        source='predefined_shortcut',
    )


vcrbench_dataset = {
    'VCRBench_8frame_nopack': _video_spec(VCRBench, dataset='VCR-Bench', nframe=8, pack=False),
    'VCRBench_16frame_nopack': _video_spec(VCRBench, dataset='VCR-Bench', nframe=16, pack=False),
    'VCRBench_32frame_nopack': _video_spec(VCRBench, dataset='VCR-Bench', nframe=32, pack=False),
    'VCRBench_64frame_nopack': _video_spec(VCRBench, dataset='VCR-Bench', nframe=64, pack=False),
    'VCRBench_1fps_nopack': _video_spec(VCRBench, dataset='VCR-Bench', fps=1.0, pack=False)
}

v2pbench_dataset = {
    'V2PBench_2frame_nopack': _video_spec(V2PBench, dataset='V2P-Bench', nframe=2, pack=False),
    'V2PBench_8frame_nopack': _video_spec(V2PBench, dataset='V2P-Bench', nframe=8, pack=False),
    'V2PBench_16frame_nopack': _video_spec(V2PBench, dataset='V2P-Bench', nframe=16, pack=False),
    'V2PBench_64frame_nopack': _video_spec(V2PBench, dataset='V2P-Bench', nframe=64, pack=False),
    'V2PBench_128frame_nopack': _video_spec(V2PBench, dataset='V2P-Bench', nframe=128, pack=False),
    'V2PBench_1fps_nopack': _video_spec(V2PBench, dataset='V2P-Bench', fps=1.0, pack=False)
}

mmbench_video_dataset = {
    'MMBench_Video_8frame_nopack': _video_spec(MMBenchVideo, dataset='MMBench-Video', nframe=8, pack=False),
    'MMBench_Video_8frame_pack': _video_spec(MMBenchVideo, dataset='MMBench-Video', nframe=8, pack=True),
    'MMBench_Video_16frame_nopack': _video_spec(MMBenchVideo, dataset='MMBench-Video', nframe=16, pack=False),
    'MMBench_Video_64frame_nopack': _video_spec(MMBenchVideo, dataset='MMBench-Video', nframe=64, pack=False),
    'MMBench_Video_64frame_pack': _video_spec(MMBenchVideo, dataset='MMBench-Video', nframe=64, pack=True),
    'MMBench_Video_1fps_nopack': _video_spec(MMBenchVideo, dataset='MMBench-Video', fps=1.0, pack=False),
    'MMBench_Video_1fps_pack': _video_spec(MMBenchVideo, dataset='MMBench-Video', fps=1.0, pack=True)
}

mvbench_dataset = {
    'MVBench_8frame': _video_spec(MVBench, dataset='MVBench', nframe=8),
    'MVBench_64frame': _video_spec(MVBench, dataset='MVBench', nframe=64),
    # MVBench not support fps, but MVBench_MP4 does
    'MVBench_MP4_8frame': _video_spec(MVBench_MP4, dataset='MVBench_MP4', nframe=8),
    'MVBench_MP4_1fps': _video_spec(MVBench_MP4, dataset='MVBench_MP4', fps=1.0),
}

tamperbench_dataset = {
    'MVTamperBench_8frame': _video_spec(MVTamperBench, dataset='MVTamperBench', nframe=8),
    'MVTamperBenchStart_8frame': _video_spec(MVTamperBench, dataset='MVTamperBenchStart', nframe=8),
    'MVTamperBenchEnd_8frame': _video_spec(MVTamperBench, dataset='MVTamperBenchEnd', nframe=8),
}

videomme_dataset = {
    'Video-MME_8frame': _video_spec(VideoMME, dataset='Video-MME', nframe=8),
    'Video-MME_64frame': _video_spec(VideoMME, dataset='Video-MME', nframe=64),
    'Video-MME_8frame_subs': _video_spec(VideoMME, dataset='Video-MME', nframe=8, use_subtitle=True),
    'Video-MME_1fps': _video_spec(VideoMME, dataset='Video-MME', fps=1.0),
    'Video-MME_0.5fps': _video_spec(VideoMME, dataset='Video-MME', fps=0.5),
    'Video-MME_0.5fps_subs': _video_spec(VideoMME, dataset='Video-MME', fps=0.5, use_subtitle=True),
}

videommev2_dataset = {
    # ── No subtitle ──
    'Video-MME-v2_64frame': _video_spec(VideoMMEv2, dataset='Video-MME-v2', nframe=64),
    'Video-MME-v2_1fps': _video_spec(VideoMMEv2, dataset='Video-MME-v2', nframe=0, fps=1.0),
    # ── Subtitle (non-interleave, concatenated as text block) ──
    'Video-MME-v2_64frame_subs': _video_spec(
        VideoMMEv2, dataset='Video-MME-v2', nframe=64, with_subtitle=True),
    'Video-MME-v2_1fps_subs': _video_spec(
        VideoMMEv2, dataset='Video-MME-v2', nframe=0, fps=1.0, with_subtitle=True),
    # ── Subtitle (interleave, timestamp-aligned between frames) ──
    'Video-MME-v2_64frame_subs_interleave': _video_spec(
        VideoMMEv2, dataset='Video-MME-v2', nframe=64,
        with_subtitle=True, subtitle_interleave=True),
    'Video-MME-v2_1fps_subs_interleave': _video_spec(
        VideoMMEv2, dataset='Video-MME-v2', nframe=0, fps=1.0,
        with_subtitle=True, subtitle_interleave=True),
    # ── Reasoning (no subtitle) ──
    'Video-MME-v2_64frame_reasoning': _video_spec(
        VideoMMEv2, dataset='Video-MME-v2', nframe=64, reasoning=True),
    # ── Reasoning + subtitle (non-interleave) ──
    'Video-MME-v2_64frame_reasoning_subs': _video_spec(
        VideoMMEv2, dataset='Video-MME-v2', nframe=64,
        reasoning=True, with_subtitle=True),
    # ── Reasoning + subtitle (interleave) ──
    'Video-MME-v2_64frame_reasoning_subs_interleave': _video_spec(
        VideoMMEv2, dataset='Video-MME-v2', nframe=64,
        reasoning=True, with_subtitle=True, subtitle_interleave=True),
    # ── Resize (no subtitle) ──
    'Video-MME-v2_64frame_resize': _video_spec(
        VideoMMEv2, dataset='Video-MME-v2', nframe=64,
        resize_target_area=448 * 448),
    'Video-MME-v2_1fps_resize': _video_spec(
        VideoMMEv2, dataset='Video-MME-v2', nframe=0, fps=1.0,
        resize_target_area=448 * 448),
    # ── Resize + subtitle ──
    'Video-MME-v2_64frame_resize_subs': _video_spec(
        VideoMMEv2, dataset='Video-MME-v2', nframe=64,
        resize_target_area=448 * 448, with_subtitle=True),
    # ── Resize + subtitle interleave ──
    'Video-MME-v2_64frame_resize_subs_interleave': _video_spec(
        VideoMMEv2, dataset='Video-MME-v2', nframe=64,
        resize_target_area=448 * 448, with_subtitle=True, subtitle_interleave=True),
    # ── Resize + reasoning ──
    'Video-MME-v2_64frame_resize_reasoning': _video_spec(
        VideoMMEv2, dataset='Video-MME-v2', nframe=64,
        resize_target_area=448 * 448, reasoning=True),
}

videommmu_dataset = {
    'VideoMMMU_8frame': _video_spec(VideoMMMU, dataset='VideoMMMU', nframe=8),
    'VideoMMMU_64frame': _video_spec(VideoMMMU, dataset='VideoMMMU', nframe=64),
    'VideoMMMU_1fps': _video_spec(VideoMMMU, dataset='VideoMMMU', fps=1.0),
    'VideoMMMU_0.5fps': _video_spec(VideoMMMU, dataset='VideoMMMU', fps=0.5),
}

longvideobench_dataset = {
    'LongVideoBench_8frame': _video_spec(LongVideoBench, dataset='LongVideoBench', nframe=8),
    'LongVideoBench_8frame_subs': _video_spec(LongVideoBench, dataset='LongVideoBench', nframe=8, use_subtitle=True),
    'LongVideoBench_64frame': _video_spec(LongVideoBench, dataset='LongVideoBench', nframe=64),
    'LongVideoBench_1fps': _video_spec(LongVideoBench, dataset='LongVideoBench', fps=1.0),
    'LongVideoBench_0.5fps': _video_spec(LongVideoBench, dataset='LongVideoBench', fps=0.5),
    'LongVideoBench_0.5fps_subs': _video_spec(LongVideoBench, dataset='LongVideoBench', fps=0.5, use_subtitle=True)
}

mlvu_dataset = {
    'MLVU_8frame': _video_spec(MLVU, dataset='MLVU', nframe=8),
    'MLVU_64frame': _video_spec(MLVU, dataset='MLVU', nframe=64),
    'MLVU_1fps': _video_spec(MLVU, dataset='MLVU', fps=1.0)
}

tempcompass_dataset = {
    'TempCompass_8frame': _video_spec(TempCompass, dataset='TempCompass', nframe=8),
    'TempCompass_64frame': _video_spec(TempCompass, dataset='TempCompass', nframe=64),
    'TempCompass_1fps': _video_spec(TempCompass, dataset='TempCompass', fps=1.0),
    'TempCompass_0.5fps': _video_spec(TempCompass, dataset='TempCompass', fps=0.5)
}

# In order to reproduce the experimental results in CGbench paper,
# use_subtitle, use_subtitle_time and use_frame_time need to be set to True.
# When measuring clue-related results, if the number of frames used is greater
# than 32, the frame capture limit will be set to 32.
# We implement the metrics long_acc, clue_acc, miou, CRR, acc@iou and rec@iou
# in the CGBench_MCQ_Grounding_Mini and CGBench_MCQ_Grounding datasets;
# the metric open-ended is implemented in the CGBench_OpenEnded_Mini and CGBench_OpenEnded datasets.
cgbench_dataset = {
    'CGBench_MCQ_Grounding_Mini_8frame_subs_subt': _video_spec(
        CGBench_MCQ_Grounding_Mini,
        dataset='CG-Bench_MCQ_Grounding_Mini',
        nframe=8,
        use_subtitle=True,
        use_subtitle_time=True
    ),
    'CGBench_OpenEnded_Mini_8frame_subs_subt_ft': _video_spec(
        CGBench_OpenEnded_Mini,
        dataset='CG-Bench_OpenEnded_Mini',
        nframe=8,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    ),
    'CGBench_MCQ_Grounding_32frame_subs': _video_spec(
        CGBench_MCQ_Grounding,
        dataset='CG-Bench_MCQ_Grounding',
        nframe=32,
        use_subtitle=True
    ),
    'CGBench_OpenEnded_8frame': _video_spec(
        CGBench_OpenEnded,
        dataset='CG-Bench_OpenEnded',
        nframe=8
    ),
    'CGBench_MCQ_Grounding_16frame_subs_subt_ft': _video_spec(
        CGBench_MCQ_Grounding,
        dataset='CG-Bench_MCQ_Grounding',
        nframe=16,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    ),
    'CGBench_OpenEnded_16frame_subs_subt_ft': _video_spec(
        CGBench_OpenEnded,
        dataset='CG-Bench_OpenEnded',
        nframe=16,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    )
}

megabench_dataset = {
    'MEGABench_core_16frame': _video_spec(MEGABench, dataset='MEGABench', nframe=16, subset_name="core"),
    'MEGABench_open_16frame': _video_spec(MEGABench, dataset='MEGABench', nframe=16, subset_name="open"),
    'MEGABench_core_64frame': _video_spec(MEGABench, dataset='MEGABench', nframe=64, subset_name="core"),
    'MEGABench_open_64frame': _video_spec(MEGABench, dataset='MEGABench', nframe=64, subset_name="open")
}

moviechat1k_dataset = {
    'moviechat1k_breakpoint_8frame': _video_spec(MovieChat1k, dataset='MovieChat1k', subset='breakpoint', nframe=8),
    'moviechat1k_global_14frame': _video_spec(MovieChat1k, dataset='MovieChat1k', subset='global', nframe=14),
    'moviechat1k_global_8frame_limit0.01': _video_spec(
        MovieChat1k, dataset='MovieChat1k', subset='global', nframe=8, limit=0.01
    )
}

vdc_dataset = {
    'VDC_8frame': _video_spec(VDC, dataset='VDC', nframe=8),
    'VDC_1fps': _video_spec(VDC, dataset='VDC', fps=1.0),
}

worldsense_dataset = {
    'WorldSense_8frame': _video_spec(WorldSense, dataset='WorldSense', nframe=8),
    'WorldSense_8frame_subs': _video_spec(WorldSense, dataset='WorldSense', nframe=8, use_subtitle=True),
    'WorldSense_8frame_audio': _video_spec(WorldSense, dataset='WorldSense', nframe=8, use_audio=True),
    'WorldSense_32frame': _video_spec(WorldSense, dataset='WorldSense', nframe=32),
    'WorldSense_32frame_subs': _video_spec(WorldSense, dataset='WorldSense', nframe=32, use_subtitle=True),
    'WorldSense_32frame_audio': _video_spec(WorldSense, dataset='WorldSense', nframe=32, use_audio=True),
    'WorldSense_1fps': _video_spec(WorldSense, dataset='WorldSense', fps=1.0),
    'WorldSense_1fps_subs': _video_spec(WorldSense, dataset='WorldSense', fps=1.0, use_subtitle=True),
    'WorldSense_1fps_audio': _video_spec(WorldSense, dataset='WorldSense', fps=1.0, use_audio=True),
    'WorldSense_0.5fps': _video_spec(WorldSense, dataset='WorldSense', fps=0.5),
    'WorldSense_0.5fps_subs': _video_spec(WorldSense, dataset='WorldSense', fps=0.5, use_subtitle=True),
    'WorldSense_0.5fps_audio': _video_spec(WorldSense, dataset='WorldSense', fps=0.5, use_audio=True)
}

qbench_video_dataset = {
    'QBench_Video_8frame': _video_spec(QBench_Video, dataset='QBench_Video', nframe=8),
    'QBench_Video_16frame': _video_spec(QBench_Video, dataset='QBench_Video', nframe=16),
}

video_mmlu_dataset = {
    'Video_MMLU_CAP_16frame': _video_spec(Video_MMLU_CAP, dataset='Video_MMLU_CAP', nframe=16),
    'Video_MMLU_CAP_64frame': _video_spec(Video_MMLU_CAP, dataset='Video_MMLU_CAP', nframe=64),
    'Video_MMLU_QA_16frame': _video_spec(Video_MMLU_QA, dataset='Video_MMLU_QA', nframe=16),
    'Video_MMLU_QA_64frame': _video_spec(Video_MMLU_QA, dataset='Video_MMLU_QA', nframe=64),
}

video_tt_dataset = {
    'Video_TT_16frame': _video_spec(VideoTT, dataset='Video-TT', nframe=16),
    'Video_TT_32frame': _video_spec(VideoTT, dataset='Video-TT', nframe=32),
    'Video_TT_64frame': _video_spec(VideoTT, dataset='Video-TT', nframe=64),
}

video_holmes_dataset = {
    'Video_Holmes_32frame': _video_spec(Video_Holmes, dataset='Video_Holmes', nframe=32),
    'Video_Holmes_64frame': _video_spec(Video_Holmes, dataset='Video_Holmes', nframe=64),
}

cg_av_counting_dataset = {
    'CG-AV-Counting_32frame': _video_spec(CGAVCounting, dataset='CG-AV-Counting', nframe=32, use_frame_time=False),
    'CG-AV-Counting_64frame': _video_spec(CGAVCounting, dataset='CG-AV-Counting', nframe=64, use_frame_time=False)
}

egoexobench_dataset = {
    'EgoExoBench_64frame': _video_spec(EgoExoBench_MCQ, dataset='EgoExoBench_MCQ', nframe=64, skip_EgoExo4D=False),  # noqa: E501
    'EgoExoBench_64frame_skip_EgoExo4D': _video_spec(EgoExoBench_MCQ, dataset='EgoExoBench_MCQ', nframe=64, skip_EgoExo4D=True)  # noqa: E501

}

revsi_dataset = {
    'revsi_16_frame': _video_spec(ReVSI, dataset='ReVSI', nframe=16),
    'revsi_32_frame': _video_spec(ReVSI, dataset='ReVSI', nframe=32),
    'revsi_64_frame': _video_spec(ReVSI, dataset='ReVSI', nframe=64),
    'revsi_all_frame': _video_spec(ReVSI, dataset='ReVSI', nframe=None),
}

dream_1k_dataset = {
    'DREAM-1K_8frame': _video_spec(DREAM, dataset='DREAM-1K', nframe=8),
    'DREAM-1K_64frame': _video_spec(DREAM, dataset='DREAM-1K', nframe=64),
    'DREAM-1K_2fps': _video_spec(DREAM, dataset='DREAM-1K', fps=2.0),
    'DREAM-1K_1fps': _video_spec(DREAM, dataset='DREAM-1K', fps=1.0),
    'DREAM-1K_0.5fps': _video_spec(DREAM, dataset='DREAM-1K', fps=0.5),
}

av_speakerbench_dataset = {
    # frame-sampled variants
    'AV-SpeakerBench_audiovisual_8frame': _video_spec(
        AVSpeakerBench, dataset='AV-SpeakerBench', nframe=8, use_audio=True
    ),
    'AV-SpeakerBench_audiovisual_16frame': _video_spec(
        AVSpeakerBench, dataset='AV-SpeakerBench', nframe=16, use_audio=True
    ),
    'AV-SpeakerBench_visual_8frame': _video_spec(
        AVSpeakerBench, dataset='AV-SpeakerBench', nframe=8, use_audio=False
    ),
    'AV-SpeakerBench_visual_16frame': _video_spec(
        AVSpeakerBench, dataset='AV-SpeakerBench', nframe=16, use_audio=False
    ),
    'AV-SpeakerBench_audio_only_8frame': _video_spec(
        AVSpeakerBench, dataset='AV-SpeakerBench', nframe=8, use_audio=True, audio_only=True
    ),
    'AV-SpeakerBench_audio_only_16frame': _video_spec(
        AVSpeakerBench, dataset='AV-SpeakerBench', nframe=16, use_audio=True, audio_only=True
    ),
    # fps-based variants
    'AV-SpeakerBench_audiovisual_1fps': _video_spec(
        AVSpeakerBench, dataset='AV-SpeakerBench', fps=1.0, use_audio=True
    ),
    'AV-SpeakerBench_visual_1fps': _video_spec(
        AVSpeakerBench, dataset='AV-SpeakerBench', fps=1.0, use_audio=False
    ),
    'AV-SpeakerBench_audio_only_1fps': _video_spec(
        AVSpeakerBench, dataset='AV-SpeakerBench', fps=1.0, use_audio=True, audio_only=True
    ),
    # shorthand aliases mapping to audiovisual
    'AV-SpeakerBench_8frame': _video_spec(
        AVSpeakerBench, dataset='AV-SpeakerBench', nframe=8, use_audio=True
    ),
    'AV-SpeakerBench_16frame': _video_spec(
        AVSpeakerBench, dataset='AV-SpeakerBench', nframe=16, use_audio=True
    ),
    'AV-SpeakerBench_1fps': _video_spec(
        AVSpeakerBench, dataset='AV-SpeakerBench', fps=1.0, use_audio=True
    ),
}

omtg_dataset = {
    "OMTGBench_1fps": _video_spec(OMTGBench, dataset="OMTGBench", fps=1.0),
    "OMTGBench_2fps": _video_spec(OMTGBench, dataset="OMTGBench", fps=2.0),
}

mvu_eval_dataset = {
    'MVU-Eval_8frame': _video_spec(MVUEval, dataset='MVU-Eval', nframe=8),
    'MVU-Eval_16frame': _video_spec(MVUEval, dataset='MVU-Eval', nframe=16),
}

VSI_FRAME_VARIANTS = [
    ("128frame", dict(nframe=128)),
    ("64frame", dict(nframe=64)),
    ("32frame", dict(nframe=32)),
    ("16frame", dict(nframe=16)),
    ("2fps", dict(fps=2.0)),
    ("1fps", dict(fps=1.0)),
]


def _build_video_variants(subsets, cls, variants=VSI_FRAME_VARIANTS):
    out = {}
    for variant in subsets:
        for suffix, params in variants:
            out[f"{variant}_{suffix}"] = _video_spec(cls, dataset=variant, **params)
    return out


# === VSI-Bench ===
vsi_subsets = VsiBench.supported_datasets()
video_vsi_dataset = _build_video_variants(vsi_subsets, VsiBench)

# === VSI-SUPER-Recall ===
vsisuper_recall_subsets = VsiSuperRecall.supported_datasets()
vsisuper_recall_dataset = _build_video_variants(vsisuper_recall_subsets, VsiSuperRecall)

# === VSI-SUPER-Count ===
vsisuper_count_subsets = VsiSuperCount.supported_datasets()
vsisuper_count_dataset = _build_video_variants(vsisuper_count_subsets, VsiSuperCount)

sitebenchvideo_dataset = {
    'SiteBenchVideo_64frame': _video_spec(SiteBenchVideo, dataset='SiteBenchVideo', nframe=64),
    'SiteBenchVideo_32frame': _video_spec(SiteBenchVideo, dataset='SiteBenchVideo', nframe=32),
    'SiteBenchVideo_1fps': _video_spec(SiteBenchVideo, dataset='SiteBenchVideo', fps=1),
}

mmsi_video_dataset = {
    # The 300 frame setting is aligned with Sufficient-Coverage policy proposed in MMSI-Video-Bench paper
    'MMSIVideoBench_300frame': _video_spec(MMSIVideoBench, dataset='MMSIVideoBench', nframe=300),
    'MMSIVideoBench_64frame': _video_spec(MMSIVideoBench, dataset='MMSIVideoBench', nframe=64),
    'MMSIVideoBench_50frame': _video_spec(MMSIVideoBench, dataset='MMSIVideoBench', nframe=50),
    'MMSIVideoBench_32frame': _video_spec(MMSIVideoBench, dataset='MMSIVideoBench', nframe=32),
    'MMSIVideoBench_1fps': _video_spec(MMSIVideoBench, dataset='MMSIVideoBench', fps=1),
}

sti_subsets = STIBench.supported_datasets()
sti_variants = [
    ("64frame", dict(nframe=64)),
    ("32frame", dict(nframe=32)),
    # The 30 frame setting is aligned with offical seting STI-Bench paper
    ("30frame", dict(nframe=30)),
    ("1fps", dict(fps=1.0)),
]
sti_dataset = _build_video_variants(sti_subsets, STIBench, sti_variants)

dsr_subsets = DSRBench.supported_datasets()
dsr_variants = [
    ("64frame", dict(nframe=64)),
    ("32frame", dict(nframe=32)),
    ("30frame", dict(nframe=30)),
    # The 1fps setting is aligned with offical seting DSR-Bench paper
    ("1fps", dict(fps=1.0)),
]
dsr_dataset = _build_video_variants(dsr_subsets, DSRBench, dsr_variants)
dataset_groups = [
    mmbench_video_dataset, mvbench_dataset, videomme_dataset, videommev2_dataset, videommmu_dataset,
    longvideobench_dataset, mlvu_dataset, tempcompass_dataset, cgbench_dataset, worldsense_dataset, tamperbench_dataset,
    megabench_dataset, qbench_video_dataset, moviechat1k_dataset, vdc_dataset, video_holmes_dataset, vcrbench_dataset,
    cg_av_counting_dataset, video_mmlu_dataset, egoexobench_dataset, dream_1k_dataset, video_tt_dataset,
    video_vsi_dataset, mvu_eval_dataset, omtg_dataset, v2pbench_dataset, av_speakerbench_dataset
]

# add by EASI team
dataset_groups += [
    sitebenchvideo_dataset, mmsi_video_dataset, vsisuper_recall_dataset, vsisuper_count_dataset,
    sti_dataset, dsr_dataset, revsi_dataset
]

PREDEFINED_DATASET_SPECS = {}

for grp in dataset_groups:
    for alias, spec in grp.items():
        PREDEFINED_DATASET_SPECS[alias] = DatasetSpec(
            dataset_alias_name=alias,
            dataset_name=spec.dataset_name,
            dataset_class_name=spec.dataset_class_name,
            build_config=spec.build_config,
            source=spec.source,
        )
