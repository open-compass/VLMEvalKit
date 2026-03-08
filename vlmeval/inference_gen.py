from weakref import ref
import collections
import torch
import torch.distributed as dist
import validators
from PIL import Image
from vlmeval.config import supported_ULM
from vlmeval.smp.tos import get_tos_prefix
from vlmeval.dataset import ImageBaseDataset
from vlmeval.smp import *

SPLITER = '<SPLITER_NEVER_USED_42>'
CLI = TosUploadClient()
is_response_err = ImageBaseDataset.is_response_err


def prediction2str(prediction):
    return SPLITER.join([x for x in prediction if len(x.strip())])


def str2prediction(x):
    segs = x.strip().split(SPLITER)
    return [s for s in segs if len(s)]


def split_think_single(prediction):
    from vlmeval.api.seed_thirdparty.postprocess import extract_and_remove_think_tags
    if isinstance(prediction, str):
        result_text, think_contents = extract_and_remove_think_tags(prediction)
        think_content = think_contents[0].strip() if len(think_contents) else ''
        ret = dict(prediction=result_text.strip(), thinking=think_content)
        if ret['thinking'] == '':
            ret['thinking'] = None
        if ret['prediction'] == '':
            ret['prediction'] = ret['thinking']
        return ret
    elif isinstance(prediction, list):
        pred = prediction2str(prediction)
        result_text, think_contents = extract_and_remove_think_tags(pred)
        think_content = think_contents[0].strip() if len(think_contents) else ''
        result_text = str2prediction(result_text)
        think_content = str2prediction(think_content) if len(think_content) else []
        if len(result_text) == 0:
            result_text = think_content
        return dict(prediction=result_text, thinking=think_content)
    else:
        raise NotImplementedError


def split_think(prediction):
    res = {}
    res['raw_prediction'] = cp.deepcopy(prediction)
    assert isinstance(prediction, list)
    split_results = [split_think_single(x) for x in prediction]
    res['prediction'] = [x['prediction'] for x in split_results]
    res['thinking'] = [x['thinking'] for x in split_results]
    return res


# For UG MODE, pred should be a LIST[LIST[SEGS]], SEGS ARE STR OR PIL IMAGE. IF FAILED, CAN ALSO BE LIST[FAIL_STR]
# For Gen MODE, pred should be a LIST[IMAGE]. IF FAILED, CAN ALSO BE LIST[FAIL_STR]
def img2urls(pred, idx, obj_prefix=DEFAULT_PREFIX, img_format='png'):
    assert isinstance(pred, list)
    new_pred = []
    jobs = []
    if not pred:
        return [], []

    if isinstance(pred[0], list):
        for item in pred:
            assert isinstance(item, list)
        for i, item in enumerate(pred):
            res = []
            for j, seg in enumerate(item):
                if isinstance(seg, Image.Image):
                    file_name = f'{idx}_{i}_{j}.{img_format}'
                    res.append(f"{TOS_URL_BASE}/{obj_prefix}/{file_name}")
                    jobs.append(dict(image=seg, obj_prefix=obj_prefix, file_name=file_name))
                elif isinstance(seg, str):
                    res.append(seg)
                else:
                    raise NotImplementedError(f'img2urls does not support type {type(seg)}')
            new_pred.append(res)
    else:
        for i, item in enumerate(pred):
            if isinstance(item, Image.Image):
                file_name = f'{idx}_{i}.{img_format}'
                new_pred.append(f"{TOS_URL_BASE}/{obj_prefix}/{file_name}")
                jobs.append(dict(image=item, obj_prefix=obj_prefix, file_name=file_name))
            elif isinstance(item, str):
                new_pred.append(item)
            else:
                raise NotImplementedError(f'img2urls does not support type {type(item)}')
    return new_pred, jobs


def img2tos_job(image, obj_prefix, file_name):
    b64 = encode_image_to_base64(image)
    return CLI.upload_base64_img_to_img_file(b64, obj_prefix, file_name)


def is_pred_instance_failed(pred):
    if validators.url(pred):
        return False
    if isinstance(pred, str):
        return is_response_err(pred)
    if pred is None:
        return True
    return False


def is_tilist(lst):
    if not isinstance(lst, list):
        return False
    has_image = sum([isinstance(x, Image.Image) for x in lst])
    has_text = sum([isinstance(x, str) for x in lst])
    if has_text and not has_image:
        logger = get_logger('GenEval')
        logger.warning(f'The list {lst} is a pure text list. ')
    return has_text


def is_pred_sample_failed(pred, num_generations=-1):
    assert isinstance(pred, list)
    if num_generations > 0 and len(pred) < num_generations:
        return True
    for item in pred:
        if is_pred_instance_failed(item):
            return True
    return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


def infer_data_api(
    model, model_name, work_dir, dataset,
    verbose=False, api_nproc=4, num_generations=-1
):
    dataset_name = dataset.dataset_name
    # The final result file that will be used for evaluation
    result_file_base = dataset.PRED_FORMAT.format(model_name=model_name, dataset_name=dataset_name)
    result_file = osp.join(work_dir, result_file_base)
    sample_record_file = get_intermediate_file_path(result_file, suffix='_sample', target_format='pkl')
    instance_record_file = get_intermediate_file_path(result_file, suffix='_instance', target_format='pkl')
    prev_file = get_intermediate_file_path(result_file, suffix='_PREV', target_format='pkl')

    sample_res = load(prev_file) if osp.exists(prev_file) else {}
    sample_res = {k: v for k, v in sample_res.items() if not is_pred_sample_failed(v)}

    data = dataset.data
    assert isinstance(data.iloc[0]['index'], str), 'The index column of the dataset must be str type.'

    # If finished, will exit without building the model
    all_finished = True
    for i in range(len(data)):
        idx = data.iloc[i]['index']
        if idx not in sample_res:
            all_finished = False
    if all_finished:
        dump(sample_res, sample_record_file)
        return model

    # Data need to be inferred
    remain_data = data[~data['index'].isin(sample_res)]
    lt, indices = len(remain_data), list(remain_data['index'])

    if hasattr(model, 'set_dump_image') and hasattr(dataset, 'dump_image'):
        model.set_dump_image(dataset.dump_image)

    structs = []
    for i in range(lt):
        item = remain_data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    # Currently num_generations is set in `run_gen.py`, so I add an assertion below
    num_generations = getattr(dataset, 'NUM_GENERATIONS', 1) if num_generations == -1 else num_generations
    # repeat the structs and indices for num_generations times
    structs = structs * num_generations
    indices = [f"{idx}:{i}" for i in range(num_generations) for idx in indices]

    instance_res = {}
    if osp.exists(instance_record_file):
        instance_res = load(instance_record_file)
        instance_res = {k: v for k, v in instance_res.items() if not is_pred_instance_failed(v)}

    structs = [s for i, s in zip(indices, structs) if i not in instance_res]
    indices = [i for i in indices if i not in instance_res]

    # API Models only generate one image at a time, the generation results is either PIL.Image or a list,
    # which looks like [text, PIL.Image, text, PIL.Image, ...]
    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(
            gen_func,
            structs,
            nproc=api_nproc,
            save=instance_record_file,
            keys=indices,
            desc=f'GenEval API Inference [{model_name} x {dataset_name}]')

    instance_res = load(instance_record_file)
    # if num_generations > 1, we need to merge the results
    ret_res = collections.defaultdict(list)
    for k, v in instance_res.items():
        new_k = ':'.join(k.split(':')[:-1])
        ret_res[new_k].append(v)

    sample_res.update(ret_res)
    for k in data['index']:
        assert k in sample_res, k
    dump(sample_res, sample_record_file)
    return model


def infer_data_opensource(
    model, model_name, work_dir, dataset,
    out_file, verbose=False, use_vllm=False, num_generations=-1
):
    dataset_name = dataset.dataset_name
    result_file_base = dataset.PRED_FORMAT.format(model_name=model_name, dataset_name=dataset_name)
    result_file = osp.join(work_dir, result_file_base)
    prev_file = get_intermediate_file_path(result_file, suffix='_PREV', target_format='pkl')

    res = load(prev_file) if osp.exists(prev_file) else {}
    # out_file is for the current rank
    if osp.exists(out_file):
        res.update(load(out_file))
    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    # The share of the current rank in the total dataset
    data = dataset.data.iloc[sheet_indices]
    data['index'] = data['index'].astype(str)
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)
    # Set the dump_image function for open source models
    if hasattr(model, 'set_dump_image') and hasattr(dataset, 'dump_image'):
        model.set_dump_image(dataset.dump_image)

    for i in tqdm(range(lt), desc=f'Infer {model_name}/{dataset_name}, Rank {rank}/{world_size}'):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(data.iloc[i])

        # If `SKIP_ERR` flag is set, the model will skip the generation if error is encountered
        if os.environ.get('SKIP_ERR', False) == '1':
            try:
                response = model.generate(message=struct, dataset=dataset_name, num_generations=num_generations)
            except RuntimeError as err:
                torch.cuda.synchronize()
                warnings.warn(f'{type(err)} {str(err)}')
                response = f'{FAIL_MSG}: {type(err)} {str(err)}'
        else:
            response = model.generate(message=struct, dataset=dataset_name, num_generations=num_generations)

        if isinstance(response, Image.Image):
            response = [response]
        elif isinstance(response, list) and is_tilist(response):
            response = [response]
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 10 == 0:
            dump(res, out_file)

    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(
    model, work_dir, model_name, dataset, verbose=False,
    api_nproc=4, num_generations=-1
):
    assert num_generations > 0, "`num_generations` should be set in `run_gen.py`"
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name

    # The final result file that will be used for evaluation
    result_file_base = dataset.PRED_FORMAT.format(model_name=model_name, dataset_name=dataset_name)
    result_file = osp.join(work_dir, result_file_base)
    sample_record_file = get_intermediate_file_path(result_file, suffix='_sample', target_format='pkl')
    instance_record_file = get_intermediate_file_path(result_file, suffix='_instance', target_format='pkl')
    prev_file = get_intermediate_file_path(result_file, suffix='_PREV', target_format='pkl')

    # Move all results generated during last eval to a `PREV.pkl` file
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            data['index'] = data['index'].astype(str)
            if 'raw_prediction' in data:
                results = {k: v for k, v in zip(data['index'], data['raw_prediction'])}
            elif 'prediction' in data:
                results = {k: v for k, v in zip(data['index'], data['prediction'])}
            else:
                raise NotImplementedError
            results = {k: v if isinstance(v, list) else eval(v) for k, v in results.items()}
            results = {k: v for k, v in results.items() if not is_pred_sample_failed(v, num_generations)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    # This will be used by open source files (multiple instances on a node)
    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)

    if getattr(model, 'is_api', False):
        assert rank == 0 and world_size == 1, (rank, world_size)
        model = infer_data_api(
            model=model, work_dir=work_dir, model_name=model_name, dataset=dataset,
            verbose=verbose, api_nproc=api_nproc, num_generations=num_generations)
    else:
        model = infer_data_opensource(
            model=model, work_dir=work_dir, model_name=model_name, dataset=dataset,
            out_file=out_file, verbose=verbose, num_generations=num_generations)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data = dataset.data
        if not osp.exists(sample_record_file):
            data_all = {}
            for i in range(world_size):
                data_all.update(load(tmpl.format(i)))
            for x in data['index']:
                assert x in data_all
            # Save Raw Predictions to Local
            dump(data_all, sample_record_file)
        else:
            data_all = load(sample_record_file)

        # Save Images to Tos
        obj_prefix = get_tos_prefix(model_name, dataset_name, project='geneval')
        prediction = []
        upload_jobs = []
        img_format = os.environ.get('DEFAULT_IMAGE_FORMAT', 'png')

        for idx in data['index']:
            pred = data_all[idx]
            new_pred, jobs = img2urls(pred=pred, idx=idx, obj_prefix=obj_prefix, img_format=img_format)
            upload_jobs.extend(jobs)
            prediction.append(new_pred)

        nproc = min(16, mp.cpu_count())
        _ = track_progress_rich(
            img2tos_job,
            upload_jobs,
            nproc=nproc,
            desc=f'Uploading Results [{model_name} x {dataset_name}]'
        )
        SPLIT_TRINK_FLAG = os.getenv('SPLIT_THINK', False)
        # NOT RUNNING IN UG MODE
        if not isinstance(prediction[0][0], list):
            SPLIT_TRINK_FLAG = False
        if SPLIT_TRINK_FLAG:
            split_results = [split_think(x) for x in prediction]
            data['prediction'] = [x['prediction'] for x in split_results]
            data['thinking'] = [x['thinking'] for x in split_results]
            data['raw_prediction'] = [x['raw_prediction'] for x in split_results]
        else:
            data['prediction'] = prediction

        dump(data, result_file)
        for i in range(world_size):
            if osp.exists(tmpl.format(i)):
                os.remove(tmpl.format(i))

    if world_size > 1:
        dist.barrier()

    for fname in [prev_file, sample_record_file, instance_record_file]:
        if osp.exists(fname):
            os.remove(fname)

    return model
