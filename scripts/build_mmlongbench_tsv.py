import argparse
import json
import os
import os.path as osp
import random
import tarfile
from copy import deepcopy

import pandas as pd


REPO_ID = 'ZhaoweiWang/MMLongBench'
RAW_ARCHIVE = '0_mmlb_data.tar.gz'
REFERENCE_TSVS = {
    'MMLongBench_32K': 'vlmevalkit/MMLongBench_32K.tsv',
    'MMLongBench_128K': 'vlmevalkit/MMLongBench_128K.tsv',
    'MMLongBench_256K': 'vlmevalkit/MMLongBench_256K.tsv',
    'MMLongBench_512K': 'vlmevalkit/MMLongBench_512K.tsv',
}

DEFAULT_DATASETS = ['MMLongBench_32K', 'MMLongBench_128K']

DATASET_LENGTH = {
    'MMLongBench_32K': 32,
    'MMLongBench_128K': 128,
    'MMLongBench_256K': 256,
    'MMLongBench_512K': 512,
}

DATASET_SUBSETS = {
    'MMLongBench_32K': {
        'vrag': 'vrag_32',
        'NIAH': 'NIAH_32',
        'ICL': 'ICL_32',
        'summ': 'summ_32',
        'documentQA': 'documentQA_32',
    },
    'MMLongBench_128K': {
        'vrag': 'vrag_128',
        'NIAH': 'NIAH_128',
        'ICL': 'ICL_128',
        'summ': 'summ_128',
        'documentQA': 'documentQA_128',
    },
    'MMLongBench_256K': {
        'vrag': 'vrag_256_sr50',
        'NIAH': 'NIAH_256_sr25',
        'ICL': 'ICL_256_sr50',
        'summ': 'summ_256_sr50',
        'documentQA': 'documentQA_256_sr50',
    },
    'MMLongBench_512K': {
        'vrag': 'vrag_512_sr50',
        'NIAH': 'NIAH_512_sr25',
        'ICL': 'ICL_512_sr50',
        'summ': 'summ_512_sr50',
        'documentQA': 'documentQA_512_sr50',
    },
}

VRAG_SOURCES = {'infoseek': 3, 'viquae': 6}
NIAH_SOURCES = {
    'vh_single': ('vh_single_test_1000_K{k}_dep6.jsonl', 'visual_haystack'),
    'vh_multi': ('vh_multi_test_1000_K{k}_dep3.jsonl', 'visual_haystack'),
    'retrieval-text': ('retrieval-text_test_K{k}_dep6.jsonl', 'text'),
    'counting-text': ('counting-text_test_K{k}_dep3.jsonl', 'text'),
    'reasoning-text': ('reasoning-text_test_K{k}_dep3.jsonl', 'text'),
    'retrieval-image': ('retrieval-image_test_K{k}_dep6.jsonl', 'image_mc'),
    'counting-image': ('counting-image_test_K{k}_dep3.jsonl', 'image_count'),
    'reasoning-image': ('reasoning-image_test_K{k}_dep6.jsonl', 'image_mc'),
}
ICL_SOURCES = ['cars196', 'sun397', 'inat2021', 'food101']
SUMM_SOURCES = ['gov', 'lexsum']
DOCQA_SOURCES = ['mmlongdoc', 'longdocurl', 'slidevqa']


def safe_extract(tar, path):
    root = osp.abspath(path)
    for member in tar.getmembers():
        target = osp.abspath(osp.join(root, member.name))
        if not (target == root or target.startswith(root + os.sep)):
            raise RuntimeError(f'Unsafe path in tar archive: {member.name}')
    tar.extractall(root)


def hf_download(filename, download_dir, token=None):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as err:
        raise ImportError(
            'huggingface_hub is required to download MMLongBench assets. '
            'Install it with `pip install huggingface_hub`, or pass local paths.'
        ) from err

    return hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type='dataset',
        local_dir=download_dir,
        token=token,
    )


def prepare_raw_data(raw_data_root, work_dir, token=None, skip_existing=False):
    if raw_data_root:
        return osp.abspath(raw_data_root)

    os.makedirs(work_dir, exist_ok=True)
    archive_path = osp.join(work_dir, RAW_ARCHIVE)
    if skip_existing and osp.exists(archive_path):
        tar_path = archive_path
    else:
        print(f'downloading {RAW_ARCHIVE} from {REPO_ID} ...')
        tar_path = hf_download(RAW_ARCHIVE, work_dir, token=token)

    extract_root = osp.join(work_dir, 'raw')
    marker = osp.join(extract_root, 'mmlb_data')
    if skip_existing and osp.exists(marker):
        return marker

    os.makedirs(extract_root, exist_ok=True)
    print(f'extracting {tar_path} ...')
    with tarfile.open(tar_path, 'r:gz') as tar:
        safe_extract(tar, extract_root)

    candidates = [
        osp.join(extract_root, 'mmlb_data'),
        osp.join(extract_root, 'MMLongBench', 'mmlb_data'),
    ]
    for candidate in candidates:
        if osp.isdir(candidate):
            return candidate
    raise FileNotFoundError(f'Cannot find mmlb_data under {extract_root}')


def prepare_reference_tsv(dataset, reference_tsv_root, work_dir, token=None, skip_existing=False):
    if reference_tsv_root:
        path = osp.join(reference_tsv_root, f'{dataset}.tsv')
        if not osp.exists(path):
            raise FileNotFoundError(f'Missing reference TSV: {path}')
        return path

    os.makedirs(work_dir, exist_ok=True)
    filename = REFERENCE_TSVS[dataset]
    local_path = osp.join(work_dir, filename)
    if skip_existing and osp.exists(local_path):
        return local_path
    print(f'downloading {filename} from {REPO_ID} ...')
    return hf_download(filename, work_dir, token=token)


def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def json_dumps(value):
    return json.dumps(value, ensure_ascii=False)


def row_key(row):
    return (row['task'], row['source_dataset'], str(row['question_id']))


def reference_id_map(reference):
    id_map = {}
    for _, row in reference.iterrows():
        key = row_key(row)
        id_map[key] = row
    return id_map


def task_ids(reference, task, source):
    data = reference[(reference['task'] == task) & (reference['source_dataset'] == source)]
    return {str(x) for x in data['question_id']}


def read_selected_jsonl(path, ids):
    if not ids:
        return []
    rows = []
    missing = set(ids)
    for sample in read_jsonl(path):
        sample_id = str(sample['id'])
        if sample_id in ids:
            rows.append(sample)
            missing.discard(sample_id)
    if missing:
        raise ValueError(f'{path} missing {len(missing)} ids, examples: {sorted(missing)[:5]}')
    return rows


def make_tsv_row(dataset, subset, task, source_dataset, question_id, question, answer, image_path,
                 extra_info, tags):
    return {
        'index': None,
        'question_id': question_id,
        'question': question,
        'answer': answer,
        'image_path': json_dumps(image_path),
        'mmlb_subset': subset,
        'task': task,
        'source_dataset': source_dataset,
        'question_type': 'open',
        'tags': json_dumps(tags),
        'extra_info': json_dumps(extra_info),
    }


def add_row(rows, key, row):
    rows.setdefault(key, []).append(row)


def convert_vrag(dataset, raw_root, reference):
    k = DATASET_LENGTH[dataset]
    subset = DATASET_SUBSETS[dataset]['vrag']
    rows = {}
    user_template = (
        'Use the given documents to write a concise and short answer to the question about the entity shown '
        'in the image. Write your answer in the following format:\n'
        'Answer: [answer]\n\n{context}\n\nQuestion: {question}'
    )
    passage_template = 'Document (Title: {title}): {text}'

    for source, dep in VRAG_SOURCES.items():
        ids = task_ids(reference, 'vrag', source)
        path = osp.join(raw_root, 'vrag', f'{source}_K{k}_dep{dep}.jsonl')
        for sample in read_selected_jsonl(path, ids):
            sample = deepcopy(sample)
            passage_text = '\n\n'.join([passage_template.format(**ctx) for ctx in sample['ctxs']])
            question = '<image token>' + sample['question']
            text = user_template.format(context=passage_text, question=question)
            assert '<image>' not in text

            question_id = str(sample.pop('id'))
            ctxs = sample.pop('ctxs')
            sample['question'] = question
            sample['text'] = text
            sample['image_list'] = [sample['image']]
            sample['dataset_name'] = source

            extra = deepcopy(sample)
            extra.pop('text')
            extra.pop('image_list')
            extra.pop('ctxs', None)
            extra['dataset_name'] = source
            if 'positive_ctxs' not in extra and ctxs:
                extra['positive_ctxs'] = []

            row = make_tsv_row(
                dataset, subset, 'vrag', source, question_id, text, str(sample['answer']),
                [sample['image']], extra, ['单图']
            )
            add_row(rows, ('vrag', source, question_id), row)
    return rows


def context_from_paragraphs(paragraphs):
    return '\n\n'.join([p['text'] if p['text'] != '<image>' else '<image token>' for p in paragraphs])


def convert_niah(dataset, raw_root, reference):
    k = DATASET_LENGTH[dataset]
    subset = DATASET_SUBSETS[dataset]['NIAH']
    rows = {}
    vh_template = (
        'You are given a set of images. Please answer the question in Yes or No based on the given images. '
        'Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}'
    )
    text_template = (
        'You are given interleaved text and images. Please answer the question based on the given text and images. '
        'Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}'
    )
    image_mc_template = (
        "You are given interleaved text and images. Please answer the question with the option's letter "
        '(A, B, etc.) based on the given text and images. Write your answer in the following format:\n'
        'Answer: [answer]\n\n{context}\n\nQuestion: {question}'
    )

    for source, (filename_tpl, kind) in NIAH_SOURCES.items():
        ids = task_ids(reference, 'NIAH', source)
        path = osp.join(raw_root, 'NIAH', filename_tpl.format(k=k))
        for sample in read_selected_jsonl(path, ids):
            sample = deepcopy(sample)
            question_id = str(sample.pop('id'))
            original_ctxs = sample.pop('ctxs')

            if kind == 'visual_haystack':
                image_list = sample['ctxs'] if 'ctxs' in sample else original_ctxs
                image_list = [x if isinstance(x, str) else x.get('image', x.get('text')) for x in image_list]
                passage_text = '\n'.join(['<image token>'] * len(image_list))
                question = sample['question']
                text = vh_template.format(context=passage_text, question=question)
            elif kind == 'text':
                image_list = sample['image_list']
                passage_text = context_from_paragraphs(original_ctxs)
                question = sample['question']
                text = text_template.format(context=passage_text, question=question)
            elif kind == 'image_count':
                image_list = sample['image_list'] + sample['needle_image_list']
                passage_text = context_from_paragraphs(original_ctxs)
                question = sample['question'].replace('<image>', '<image token>')
                text = text_template.format(context=passage_text, question=question)
                for ctx in sample.get('positive_ctxs', []):
                    if ctx.get('type') == 'image':
                        ctx['text'] = ctx['text'].replace('<image>', '<image token>')
            elif kind == 'image_mc':
                passage_text = context_from_paragraphs(original_ctxs)
                question = sample['question']
                question += ''.join([f'\n{chr(i + ord("A"))}. <image token>' for i in range(len(sample['choices_image']))])
                text = image_mc_template.format(context=passage_text, question=question)
                image_list = sample['image_list'] + sample['choices_image']
                for ctx in sample.get('positive_ctxs', []):
                    if ctx.get('type') == 'image':
                        ctx['text'] = ctx['text'].replace('<image>', '<image token>')
            else:
                raise ValueError(f'Unknown NIAH kind: {kind}')

            assert '<image>' not in text
            sample['text'] = text
            sample['image_list'] = image_list
            sample['dataset_name'] = source
            if kind in {'image_count', 'image_mc'}:
                sample['question'] = question

            extra = deepcopy(sample)
            extra.pop('text')
            extra.pop('image_list')
            row = make_tsv_row(
                dataset, subset, 'NIAH', source, question_id, text, str(sample['answer']),
                image_list, extra, ['多图']
            )
            add_row(rows, ('NIAH', source, question_id), row)
    return rows


def convert_icl(dataset, raw_root, reference):
    k = DATASET_LENGTH[dataset]
    subset = DATASET_SUBSETS[dataset]['ICL']
    rows = {}
    rng = random.Random(42)
    user_template = (
        'You need to recognize entities in images. Use the provided mapping from the image to label to assign '
        'a label to the test image. Only output "label: {{label}}" and nothing else.\n\n'
        'Training examples:\n{context}\n\nNow classify this image: {question}'
    )
    item_template = '<image token>\nlabel: {label}'

    for source in ICL_SOURCES:
        ids = task_ids(reference, 'ICL', source)
        if not ids:
            continue
        path = osp.join(raw_root, 'ICL', f'{source}_K{k}.json')
        data = load_json(path)
        exemplar_by_domain = {domain: data_dict['exemplar_list'] for domain, data_dict in data.items()}
        test_examples = [
            {'domain': domain, 'example': example}
            for domain, data_dict in data.items()
            for example in data_dict['test_example']
            if str(example['id']) in ids
        ]
        found = {str(x['example']['id']) for x in test_examples}
        missing = ids - found
        if missing:
            raise ValueError(f'{path} missing {len(missing)} ids, examples: {sorted(missing)[:5]}')

        for sample in test_examples:
            domain = sample['domain']
            example = sample['example']
            sampled_exemplar = deepcopy(rng.choice(exemplar_by_domain[domain]))
            for round_items in sampled_exemplar:
                rng.shuffle(round_items)
            exemplars = [item for round_items in sampled_exemplar for item in round_items]

            question = '<image token>'
            context = '\n\n'.join([item_template.format(label=item['id']) for item in exemplars])
            text = user_template.format(context=context, question=question)
            image_list = [item['image'] for item in exemplars] + [example['image']]

            question_id = str(example['id'])
            extra = {
                'domain': domain,
                'example': example,
                'text': text,
                'image_list': image_list,
                'answer': example['answer'],
                'question': question,
                'dataset_name': source,
            }
            extra.pop('text')
            extra.pop('image_list')
            row = make_tsv_row(
                dataset, subset, 'ICL', source, question_id, text, str(example['answer']),
                image_list, extra, ['多图']
            )
            add_row(rows, ('ICL', source, question_id), row)
    return rows


def convert_summ(dataset, raw_root, reference):
    k = DATASET_LENGTH[dataset]
    subset = DATASET_SUBSETS[dataset]['summ']
    rows = {}
    gov_template = (
        'You are given a government report from U.S. Government Accountability Office (GAO), and you are tasked '
        'to summarize the report. Write a concise summary (around 550 words) organized in multiple paragraphs. '
        'Where applicable, the summary should contain a short description of why GAO did this study, what GAO found, '
        'and what GAO recommends.\n\nGovernment Report:\n{context}\n\nNow please summarize the report.'
    )
    lexsum_template = (
        'You are given the legal documents in a civil rights lawsuit, and you are tasked to summarize the case. '
        'Write a concise summary of one paragraph (200 to 250 words). The summary should contain a short description '
        'of the background, the parties involved, and the outcomes of the case.\n\nLegal documents:\n{context}\n\n'
        'Now please summarize the case.'
    )
    item_template = 'Document {doc_id:.15} (page {page_id}): <image token>'

    for source in SUMM_SOURCES:
        ids = task_ids(reference, 'summ', source)
        path = osp.join(raw_root, 'summ', f'{source}_K{k}.jsonl')
        for sample in read_selected_jsonl(path, ids):
            sample = deepcopy(sample)
            question_id = str(sample.pop('id'))
            page_prompt_list = [
                item_template.format(
                    doc_id=image_path.split('/')[-2],
                    page_id=image_path.split('page')[1].split('.')[0],
                )
                for image_path in sample['image_list']
            ]
            passage_text = '\n\n'.join(page_prompt_list)
            if source == 'gov':
                text = gov_template.format(context=passage_text)
                answer = '\n\n'.join([
                    aspect['section_title'] + ':\n' + '\n'.join(aspect['paragraphs'])
                    for aspect in sample['summary']
                ])
            elif source == 'lexsum':
                text = lexsum_template.format(context=passage_text)
                answer = sample['summary']
            else:
                raise ValueError(f'Unknown summarization source: {source}')

            sample['text'] = text
            sample['answer'] = answer
            sample['dataset_name'] = source
            extra = deepcopy(sample)
            extra.pop('text')
            extra.pop('image_list')
            row = make_tsv_row(
                dataset, subset, 'summ', source, question_id, text, answer,
                sample['image_list'], extra, ['多图']
            )
            add_row(rows, ('summ', source, question_id), row)
    return rows


def convert_docqa(dataset, raw_root, reference):
    k = DATASET_LENGTH[dataset]
    subset = DATASET_SUBSETS[dataset]['documentQA']
    rows = {}
    user_template = (
        "You are given a document with text and images, and a question. Answer the question as concisely as you can, "
        "using a single phrase or sentence if possible. If the question cannot be answered based on the information "
        "in the article, write 'Not answerable.' Write your answer in the following format:\n"
        'Answer: [answer]\n\n{context}\n\nQuestion: {question}'
    )
    item_template = 'Document {doc_id:.15}: <image token>'

    for source in DOCQA_SOURCES:
        ids = task_ids(reference, 'documentQA', source)
        path = osp.join(raw_root, 'documentQA', f'{source}_K{k}.jsonl')
        for sample in read_selected_jsonl(path, ids):
            sample = deepcopy(sample)
            question_id = str(sample.pop('id'))
            image_list = sample.pop('page_list')
            page_prompt_list = [
                item_template.format(doc_id=image_path.split('/')[-2])
                for image_path in image_list
            ]
            passage_text = '\n\n'.join(page_prompt_list)
            question = (
                'Based on Document {doc_id:.15}, answer the following question. '.format(doc_id=sample['doc_name'])
                + sample['question']
            )
            text = user_template.format(context=passage_text, question=question)

            sample['text'] = text
            sample['image_list'] = image_list
            sample['question'] = question
            sample['dataset_name'] = source
            extra = deepcopy(sample)
            extra.pop('text')
            extra.pop('image_list')
            row = make_tsv_row(
                dataset, subset, 'documentQA', source, question_id, text, sample['answer'],
                image_list, extra, ['多图']
            )
            add_row(rows, ('documentQA', source, question_id), row)
    return rows


def build_dataset(dataset, raw_root, reference):
    converters = [convert_vrag, convert_niah, convert_icl, convert_summ, convert_docqa]
    generated = {}
    for converter in converters:
        generated.update(converter(dataset, raw_root, reference))

    ordered_rows = []
    missing = []
    cursor = {}
    for _, ref_row in reference.iterrows():
        key = row_key(ref_row)
        variants = generated.get(key)
        pos = cursor.get(key, 0)
        if variants is None or pos >= len(variants):
            missing.append(key)
            continue
        cursor[key] = pos + 1
        row = variants[pos]
        row = row.copy()
        row['index'] = ref_row['index']
        # ICL prompts include randomly selected few-shot exemplars. The released
        # TSVs already publish this split-specific exemplar order, so preserve it
        # instead of re-sampling and producing a different public benchmark.
        if key[0] == 'ICL':
            row['question'] = ref_row['question']
            row['image_path'] = ref_row['image_path']
        ordered_rows.append(row)
    if missing:
        raise ValueError(f'Missing {len(missing)} generated rows, examples: {missing[:5]}')
    return pd.DataFrame(ordered_rows, columns=list(reference.columns))


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Build MMLongBench VLMEvalKit TSVs from the public MMLongBench raw format. '
            'The released VLMEvalKit TSVs are used as the public split ID/order reference. '
            'The public raw archive contains the 32K/128K raw files; 256K/512K can be built '
            'only when matching extended raw files are provided via --raw-data-root.'
        )
    )
    parser.add_argument(
        '--raw-data-root',
        default=None,
        help='Path to extracted mmlb_data. If omitted, 0_mmlb_data.tar.gz is downloaded from Hugging Face.',
    )
    parser.add_argument(
        '--reference-tsv-root',
        default=None,
        help='Directory containing released MMLongBench_*.tsv files. If omitted, they are downloaded from Hugging Face.',
    )
    parser.add_argument(
        '--output-root',
        required=True,
        help='Directory where rebuilt TSV files will be written.',
    )
    parser.add_argument(
        '--work-dir',
        default=None,
        help='Download/extraction cache directory. Defaults to <output-root>/mmlongbench_downloads.',
    )
    parser.add_argument(
        '--dataset',
        nargs='+',
        default=DEFAULT_DATASETS,
        choices=list(DATASET_LENGTH),
        help='Datasets to build. Defaults to public-raw-archive splits: MMLongBench_32K MMLongBench_128K.',
    )
    parser.add_argument(
        '--token',
        default=os.environ.get('HF_TOKEN'),
        help='Optional Hugging Face token. Defaults to HF_TOKEN.',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Reuse local downloads/extractions when present.',
    )
    args = parser.parse_args()

    output_root = osp.abspath(args.output_root)
    work_dir = osp.abspath(args.work_dir or osp.join(output_root, 'mmlongbench_downloads'))
    os.makedirs(output_root, exist_ok=True)

    raw_root = prepare_raw_data(args.raw_data_root, work_dir, token=args.token, skip_existing=args.skip_existing)
    print(f'using raw data root: {raw_root}')

    for dataset in args.dataset:
        ref_path = prepare_reference_tsv(
            dataset,
            args.reference_tsv_root,
            work_dir,
            token=args.token,
            skip_existing=args.skip_existing,
        )
        reference = pd.read_csv(ref_path, sep='\t')
        data = build_dataset(dataset, raw_root, reference)
        out_file = osp.join(output_root, f'{dataset}.tsv')
        data.to_csv(out_file, sep='\t', index=False)
        print(f'wrote {out_file}: {len(data)} rows')


if __name__ == '__main__':
    main()
