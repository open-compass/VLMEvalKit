from ..image_mcq import ImageMCQDataset


MMMB_URLS = {
    'MMMB_ar': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_ar.tsv',
    'MMMB_cn': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_cn.tsv',
    'MMMB_en': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_en.tsv',
    'MMMB_pt': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_pt.tsv',
    'MMMB_ru': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_ru.tsv',
    'MMMB_tr': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_tr.tsv',
}

MTL_MMBench_URLS = {
    'MMBench_dev_ar': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_ar.tsv',
    'MMBench_dev_cn': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_cn.tsv',
    'MMBench_dev_en': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_en.tsv',
    'MMBench_dev_pt': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_pt.tsv',
    'MMBench_dev_tr': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_tr.tsv',
    'MMBench_dev_ru': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_ru.tsv',
}

MMMB_MD5 = {
    'MMMB_ar': 'f3a18b6385f1d9701840aa42de27aead', 'MMMB_cn': '13ed82fa89730037292fcaa27f08f430',
    'MMMB_en': '1cd781a71ec5a2983c090b84105d6a01', 'MMMB_pt': '548ea2b3bb2da991790386f0015d30d1',
    'MMMB_ru': 'ce1cc8a0533425ab0d86b326ebfc2984', 'MMMB_tr': '0733739d43090327975294292bc5cd67'
}

MTL_MMBench_MD5 = {
    'MMBench_dev_ar': '4271b4a0d0200e1a86380a878e0d64a4', 'MMBench_dev_cn': '2ed5135326fed02c8e51ea50dda8222f',
    'MMBench_dev_en': 'd9ab776fc018b3d45785e9a5c23431c2', 'MMBench_dev_pt': '4ddfbcd27ef12444b908c03831cd0295',
    'MMBench_dev_tr': '4fab39d501389d3d6cc90264bb708f11', 'MMBench_dev_ru': '5ba1171ff2e68f80637bf78349e402a5'
}


class MMMB(ImageMCQDataset):
    TYPE = 'MCQ'
    DATASET_URL = MMMB_URLS
    DATASET_MD5 = MMMB_MD5


class MTL_MMBench(ImageMCQDataset):
    TYPE = 'MCQ'
    DATASET_URL = MTL_MMBench_URLS
    DATASET_MD5 = MTL_MMBench_MD5
