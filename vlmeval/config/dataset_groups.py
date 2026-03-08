DATASET_GROUPS_W_FORMAT = {
    'l1': [
        ('MMVet', 'gpt-4-turbo_score.csv'), ('MMMU_DEV_VAL', 'acc.csv'),
        ('MathVista_MINI', 'gpt-4-turbo_score.csv'), ('HallusionBench', 'score.csv'),
        ('OCRBench', 'score.json'), ('AI2D_TEST', 'acc.csv'), ('MMStar', 'acc.csv'),
        ('MMBench_V11', 'acc.csv'), ('MMBench_CN_V11', 'acc.csv')
    ],
    'l2': [
        ('MME', 'score.csv'), ('LLaVABench', 'score.csv'), ('RealWorldQA', 'acc.csv'),
        ('MMBench', 'acc.csv'), ('MMBench_CN', 'acc.csv'), ('CCBench', 'acc.csv'),
        ('SEEDBench_IMG', 'acc.csv'), ('COCO_VAL', 'score.json'), ('POPE', 'score.csv'),
        ('ScienceQA_VAL', 'acc.csv'), ('ScienceQA_TEST', 'acc.csv'), ('MMT-Bench_VAL', 'acc.csv'),
        ('SEEDBench2_Plus', 'acc.csv'), ('BLINK', 'acc.csv'), ('MTVQA_TEST', 'acc.json'),
        ('Q-Bench1_VAL', 'acc.csv'), ('A-Bench_VAL', 'acc.csv'), ('R-Bench-Dis', 'acc.csv'),
    ],
    'l3': [
        ('OCRVQA_TESTCORE', 'acc.csv'), ('TextVQA_VAL', 'acc.csv'),
        ('ChartQA_TEST', 'acc.csv'), ('DocVQA_VAL', 'acc.csv'), ('InfoVQA_VAL', 'acc.csv'),
        ('SEEDBench2', 'acc.csv')
    ],
    'live': [
        ('LiveMMBench_VQ_circular', 'acc.csv'), ('LiveMMBench_Spatial_circular', 'acc.csv'),
        ('LiveMMBench_Reasoning_circular', 'acc.csv'), ('LiveMMBench_Infographic', 'acc.csv'),
        ('LiveMMBench_Perception', 'acc.csv'), ('LiveMMBench_Creation', 'merged_score.json'),
    ],
    'math': [
        ('MathVision', 'score.csv'), ('MathVerse_MINI_Vision_Only', 'score.csv'),
        ('DynaMath', 'score.csv'), ('WeMath', 'score.csv'), ('LogicVista', 'score.csv'),
        ('MathVista_MINI', 'gpt-4-turbo_score.csv'),
    ],
    'spatial': [
        ('LEGO_circular', 'acc_all.csv'), ('BLINK_circular', 'acc_all.csv'), ('MMSIBench_circular', 'acc_all.csv'),
        ('Spatial457', 'score.json'), ('3DSRBench', 'acc_all.csv')
    ]
}

DATASET_GROUPS = {
    k: [x[0] for x in v] for k, v in DATASET_GROUPS_W_FORMAT.items()
}

OBM25_GROUPS = {
    'Reasoning': [
        # 'HiPHO',
        'MicroVQA', 
        # 'HLE',
        'SFE',
        'MSEarthMCQ',
        # 'VPCT',
        'ZEROBench', 
        'ZEROBench_sub',
        'MathVision', 
        'EMMA', 
        'EMMA_COT',
        'EMMA_MINI',
        # 'ScienceQA_VAL',
        # 'ScienceQA_TEST', 
        'MathVista_MINI', 
        'CMMMU_VAL', 
        'MMMU_DEV_VAL', 
        'MMMU_Pro_10c',
        'MMMU_Pro_V',
        'MMMU_Pro_10c_COT',
        'MMMU_Pro_V_COT',
        'WeMath', 
        'DynaMath',
        'LogicVista', 
        'VisuLogic',
        'PuzzleWorld',
    ], 
    'GeneraVQA': [
        # 'CodeCognition', 
        'VLMBias', 
        'BLINK', 
        'MMStar', 
        'MMBench_DEV_EN_V11', 
        'MMBench_DEV_CN_V11',
        # 'MME', 
        'VStarBench', 
        'VLMBlind', 
        'SimpleVQA',
        'CountBenchQA',
        # 'FSC147', 
        # 'XLRS-Bench-lite',
        'MME-RealWorld-Lite', 
        'RealWorldQA'
    ], 
    'InfoGraphics': [
        'OmniDocBench', 
        'TextVQA_VAL',
        'AI2D_TEST',
        'AI2D_TEST_NO_MASK',
        'ChartQA_TEST',
        'ChartQAPro',
        'InfoVQA_VAL',
        'DocVQA_VAL',
        'OCRBench',
        # 'OCRBench_v2', 
        # 'CharXiv_descriptive_val',
        # 'CharXiv_reasoning_val'   
    ],
    'Spatial': [
        # 'VSIBench', 
        # 'AllAngle', 
        'MMSIBench_circular', 
        # 'ERQA'
    ],
    'ML&Hal': [
        # 'Vibe-Eval',
        'HallusionBench', 
        'MMVP',
        # 'IIW',
        'MTVQA_TEST'
    ],
    'Code': [
        # 'Plot2Code', 
        # 'ChartMimic_v2_direct', 
        # 'Design2Code', 
        # 'FlameReact'
    ], 
    'LongContext': [
        'DUDE_MINI',
        # 'LongDocURL',
        # 'MMLongBench'
    ], 
    'MI&Others': [
        # 'GenAI',
        # 'IntPhysics', 
        'MUIRBench', 
        'VLM2Bench', 
        # 'KCMMBench'
    ]
}

SEED20_GROUPS = {
    'Reasoning_Math': [
        'MathVista_MINI', 'MathVision', 'DynaMath', 'MathKangaroo',
        'MathVerse_MINI_Vision_Only', 'WeMath', 'MathCanvas'
    ], 
    'Reasoning_Stem': [
        'MMMU_DEV_VAL',  'MMMU_Pro_10c', 'MMMU_Pro_V', 
        'EMMA', 'SFE', 'HiPhO', 'MedXpertQA_MM_test', 'MicroVQA', 
        'XLRS-Bench-lite', 'PhyX_mini_MC', 'PhyX_mini_OE', 
        # Below are benchmarks not on the release candidate version
        'MaCBench', 'VQARAD', 'MSEarthMCQ', 
    ],
    'Reasoning_Puzzle': [
        'LogicVista', 'VPCT', 'ZEROBench_fix', 'ZEROBench_sub_fix', 
        'PuzzleWorld', 'ArcAGI1-Image', 'ArcAGI2-Image', 'VisuLogic'
    ],
    'Perception': [
        'VLMBias', 'VLMBlind', 'RealWorldQA', 'BabyVision'
    ],
    'GeneraVQA': [
        'SimpleVQA', 'HallusionBench', 'MMStar', 'MMBench_DEV_EN_V11', 
        'MMBench_DEV_CN_V11', 'MMVP', 'MUIRBench', 'MTVQA_TEST', 
        'VisFactor', 'WorldVQA', 'VibeEval'
    ],
    'InfoGraphics': [
        'AI2D_TEST', 'ChartQAPro', 'OCRBench_v2', 'CharXiv_descriptive_val',
        'CharXiv_reasoning_val',
    ], 
    'Counting': [
        'CountBenchQA', 'FSC147', 'PointBench_SEED',
    ],
    'Spatial': [
        'BLINK', 'MMSIBench_circular', 'TreeBench', 'RefSpatial_Bench',
        'CV-Bench-2D', 'CV-Bench-3D', 'ERQA'
    ],
}

DATASET_GROUPS['OBM25'] = []
for k in OBM25_GROUPS:
    DATASET_GROUPS['OBM25'].extend(OBM25_GROUPS[k])
    DATASET_GROUPS['OBM25_' + k] = OBM25_GROUPS[k]

DATASET_GROUPS['SEED20'] = []
for k in SEED20_GROUPS:
    DATASET_GROUPS['SEED20'].extend(SEED20_GROUPS[k])
    DATASET_GROUPS['SEED20_' + k] = SEED20_GROUPS[k]

DATASET_GROUPS['GEN_ALL'] = [
    'ImgGenEvalV4', 'ImgGenEvalV5', 'ImgEditEval', 'UToolEval',
    'GenEvalPP', 'GenExam', 'ImagineBench', 'RISEBench', 'WISE', 'OmniContext',
]

DATASET_GROUPS['GENUG_ALL'] = DATASET_GROUPS['GEN_ALL'] + ['Unifyv0', 'VTBench']