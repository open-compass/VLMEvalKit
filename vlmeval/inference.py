import torch
import torch.distributed as dist
import multiprocessing as mp
from vlmeval.config import build_model
from vlmeval.smp import *
from vlmeval.dataset import ImageBaseDataset


is_response_err = ImageBaseDataset.is_response_err


def omni_remove_think_tag(s):
    from vlmeval.api.seed_thirdparty.postprocess import extract_and_remove_think_tags
    if isinstance(s, str):
        pred, think = extract_and_remove_think_tags(s)
        pred = pred.strip() if len(pred.strip()) else think[0]
        return dict(raw_prediction=s, prediction=pred, thinking=think)
    elif isinstance(s, list):
        ret = [omni_remove_think_tag(x) for x in s]
        results = {}
        for k in ['raw_prediction', 'prediction', 'thinking']:
            results[k] = [x[k] for x in ret]
        return results
    else:
        raise NotImplementedError(f'omni_remove_think_tag not implemented for {type(s)}')


def validate_prediction(prediction):
    if prediction is None or pd.isna(prediction) or prediction == '':
        return False
    if is_response_err(prediction):
        return False
    return True


def convert_ug_prediction(lst):
    # Here we use small resolution to save space.
    if isinstance(lst, str):
        return lst
    new_pred = []
    for item in lst:
        if isinstance(item, str):
            new_pred.append(item)
        elif isinstance(item, Image.Image):
            b64 = encode_image_to_base64(item, target_size=512, fmt='JPEG')
            new_pred.append(f'data:image/jpeg;base64,{b64}')
        else:
            raise NotImplementedError
    return new_pred


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, out_file=None, api_nproc=4, verbose=False):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data

    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        try:
            res.update(load(out_file))
        except:
            os.remove(out_file)
    res = {k: v for k, v in res.items() if not is_response_err(v)}
    indices = list(data['index'])
    missing = [i for i in indices if i not in res]
    if len(missing) == 0:
        dump(res, out_file)
        return

    # subset that needs inference
    data = data[data['index'].isin(set(missing))]

    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)
    indices = list(data['index'])

    if dataset.with_inferencer():
        func = dataset.inference
        jobs = [dict(model=model, sample=data.iloc[i]) for i in range(len(data))]
        track_progress_rich(
            func, jobs, nproc=api_nproc, save=out_file, keys=indices,
            desc=f'API Inference [{model_name} on {dataset_name}, dataset inferencer]')
        return

    structs = []
    if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
        assert hasattr(model, 'build_prompt')
        func = model.build_prompt
        jobs = [dict(line=data.iloc[i], dataset=dataset_name) for i in range(len(data))]
    else:
        func = dataset.build_prompt
        jobs = [dict(line=data.iloc[i]) for i in range(len(data))]

    data_nproc = min(mp.cpu_count(), 32)
    structs = track_progress_rich(
        func, jobs, nproc=data_nproc, desc=f'Building Prompt [{dataset_name}]')

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(
            gen_func,
            structs,
            nproc=api_nproc,
            save=out_file,
            keys=indices,
            desc=f'API Inference [{model_name} on {dataset_name}]')


def _build_infer_model(model, model_name, use_vllm=False):
    if not isinstance(model, str):
        return model
    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
    ):
        kwargs = {'use_vllm': use_vllm}

    ws_bak = os.environ.pop('WORLD_SIZE', None)
    model = build_model(model_name, **kwargs)
    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak
    return model


def infer_data_opensource(model, model_name, work_dir, dataset, out_file, verbose=False):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        try:
            res.update(load(out_file))
        except:
            os.remove(out_file)

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]
    data_indices_set = set(data_indices)
    # We only keep those results in data_indices
    res = {k: v for k, v in res.items() if k in data_indices_set}
    dump(res, out_file)

    all_finished = all([idx in res for idx in data_indices])
    if all_finished:
        return

    data = data[~data['index'].isin(res)]
    lt = len(data)

    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    for i in tqdm(range(lt), desc=f'Infer {model_name}/{dataset_name}, Rank {rank}/{world_size}'):
        idx = data.iloc[i]['index']

        if dataset.with_inferencer():
            try:
                response = dataset.inference(model, data.iloc[i])
            except RuntimeError as err:
                if os.environ.get('SKIP_ERR', False) == '1':
                    torch.cuda.synchronize()
                    warnings.warn(f'{type(err)} {str(err)}')
                    response = f'{FAIL_MSG}: {type(err)} {str(err)}'
                else:
                    raise err
        else:
            if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
                struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
            else:
                struct = dataset.build_prompt(data.iloc[i])
            try:
                response = model.generate(message=struct, dataset=dataset_name)
            except RuntimeError as err:
                if os.environ.get('SKIP_ERR', False) == '1':
                    torch.cuda.synchronize()
                    warnings.warn(f'{type(err)} {str(err)}')
                    response = f'{FAIL_MSG}: {type(err)} {str(err)}'
                else:
                    raise err
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 10 == 0 or lt - i < 10:
            dump(res, out_file)


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(
    model, work_dir, model_name, dataset, verbose=False, api_nproc=4, use_vllm=False
):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    # 使用环境变量控制的文件格式
    result_file = get_pred_file_path(work_dir, model_name, dataset_name, use_env_format=True)

    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            # breakpoint()
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            results = {k: v for k, v in results.items() if validate_prediction(v)}
            dump(results, prev_file)
            time.sleep(2)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)
    if osp.exists(result_file) and osp.exists(out_file):
        os.remove(out_file)

    model = _build_infer_model(model, model_name=model_name, use_vllm=use_vllm)
    is_api = getattr(model, 'is_api', False)
    if is_api:
        out_file = get_intermediate_file_path(result_file, '_supp', 'pkl')
        infer_data_api(
            model=model,
            model_name=model_name,
            work_dir=work_dir,
            dataset=dataset,
            out_file=out_file,
            api_nproc=api_nproc,
            verbose=verbose,
        )
    else:
        infer_data_opensource(
            model=model,
            model_name=model_name,
            work_dir=work_dir,
            dataset=dataset,
            out_file=out_file,
            verbose=verbose)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        if is_api:
            data_all = load(out_file)
        else:
            for i in range(world_size):
                data_all.update(load(tmpl.format(i)))

        data = cp.deepcopy(dataset.data)
        for x in data['index']:
            assert x in data_all

        if hasattr(model, 'EXPERTISE') and 'TI2TI' in model.EXPERTISE:
            # which means you are evaluating an UG model
            prediction = [data_all[x] for x in data['index']]
            ug_prediction = track_progress_rich(
                convert_ug_prediction, [dict(lst=x) for x in prediction], nproc=16)
            data['ug_prediction'] = ug_prediction

            def extract_text_response(lst):
                if isinstance(lst, str):
                    return lst
                return '\n'.join([x for x in lst if isinstance(x, str)])
            data['prediction'] = [extract_text_response(x) for x in prediction]
        else:
            prediction = [data_all[x] for x in data['index']]
            if all([isinstance(x, dict) for x in prediction]):
                # Handling scenarios that `prediction` is generated with `stats` in a dict
                assert 'response' in prediction[0] and 'stats' in prediction[0], prediction[0]
                data['prediction'] = [str(item['response']) for item in prediction]
                data['stats'] = [item['stats'] for item in prediction]
            else:
                data['prediction'] = [str(item) for item in prediction]

        if os.getenv('SPLIT_THINK', False):
            # Prediction is Text Only now
            prediction = list(data['prediction'])
            print(f'Prediction format: {os.getenv("SPLIT_THINK")},splitting func: {extract_and_remove_think_tags}')

            if dataset.TYPE == 'MT':
                prediction = [toliststr(x) for x in prediction]
            results = [omni_remove_think_tag(x) for x in prediction]
            for k in ['raw_prediction', 'prediction', 'thinking']:
                data[k] = [x[k] for x in results]

        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        rm_files = list([out_file, prev_file] + [tmpl.format(i) for i in range(world_size)])
        for f in rm_files:
            if osp.exists(f):
                os.remove(f)
    if world_size > 1:
        dist.barrier()
    return model
