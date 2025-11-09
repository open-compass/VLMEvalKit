import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *

FAIL_MSG = 'Failed to obtain answer via API.'
NOT_USE_SIBENCH_PROMPT = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, actual_dataset_name, index_set=None, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name) and NOT_USE_SIBENCH_PROMPT:
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    out_file = f'{work_dir}/{model_name}_{actual_dataset_name}_supp.pkl'

    # To reuse records in MMBench_V11
    if dataset_name in ['MMBench', 'MMBench_CN']:
        v11_pred = f'{work_dir}/{model_name}_{actual_dataset_name}_V11.xlsx'
        if osp.exists(v11_pred):
            try:
                reuse_inds = load('http://opencompass.openxlab.space/utils/mmb_reuse.pkl')
                data = load(v11_pred)
                ans_map = {x: y for x, y in zip(data['index'], data['prediction']) if x in reuse_inds}
                dump(ans_map, out_file)
            except Exception as err:
                print(type(err), err)

    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
    os.remove(out_file)
    return res


def infer_data(model, model_name, work_dir, dataset, actual_dataset_name, data_base, out_file, verbose=False, api_nproc=4, use_vllm=False):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{actual_dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
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

    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
    ):
        kwargs = {'use_vllm': use_vllm}

    # (25.06.05) In newer version of transformers (after 4.50), with device_map='auto' and torchrun launcher,
    # Transformers automatically adopt TP parallelism, which leads to compatibility problems with VLMEvalKit
    # (In VLMEvalKit, we use torchrun to launch multiple model instances on a single node).
    # To bypass this problem, we unset `WORLD_SIZE` before building the model to not use TP parallel.
    ws_bak = os.environ.pop('WORLD_SIZE', None)
    model = supported_VLM[model_name](**kwargs) if isinstance(model, str) else model
    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak

    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            actual_dataset_name=actual_dataset_name,
            index_set=set(indices),
            api_nproc=api_nproc)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model
    else:
        model.set_dump_image(dataset.dump_image)
    
    assert not getattr(dataset, 'pack', False), 'Current model not supported pack mode!'
    if 'megabench' in dataset_name.lower() and 'llava_onevision' in model_name:
        print(
            'LLaVA-OneVision does not support Megabench dataset as video dataset, '
            'will set its VIDEO_LLM to False to enable multi-image input for video.'
        )
        setattr(model, 'VIDEO_LLM', False)

    for i in tqdm(range(lt), desc=f'Infer {model_name}/{actual_dataset_name}, Rank {rank}/{world_size}'):
        idx = data.iloc[i]['index']
        if idx in res:
            continue
        
        if data.iloc[i]['input_type'] in ['image', 'multi-view']:
            if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name) and NOT_USE_SIBENCH_PROMPT:
                struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
            else:
                struct = dataset.build_prompt(data.iloc[i], data_base=data_base)

            # If `SKIP_ERR` flag is set, the model will skip the generation if error is encountered
            if os.environ.get('SKIP_ERR', False) == '1':
                FAIL_MSG = 'Failed to obtain answer'
                try:
                    response = model.generate(message=struct, dataset=dataset_name)
                except RuntimeError as err:
                    torch.cuda.synchronize()
                    warnings.warn(f'{type(err)} {str(err)}')
                    response = f'{FAIL_MSG}: {type(err)} {str(err)}'
            else:
                response = model.generate(message=struct, dataset=dataset_name)

        elif data.iloc[i]['input_type'] == 'video':
            if getattr(model, 'nframe', None) is not None and getattr(model, 'nframe', 0) > 0:
                if dataset.nframe > 0:
                    if getattr(model, 'nframe', 0) != dataset.nframe:
                        print(f'{model_name} is a video-llm model, nframe is set to {dataset.nframe}, not using default')
                        setattr(model, 'nframe', dataset.nframe)
                elif getattr(model, 'fps', 0) == 0:
                    raise ValueError(f'fps is not suitable for {model_name}')
                else:
                    setattr(model, 'nframe', None)
            if getattr(model, 'fps', None) is not None and getattr(model, 'fps', 0) > 0:
                if dataset.fps > 0:
                    if getattr(model, 'fps', 0) != dataset.fps:
                        print(f'{model_name} is a video-llm model, fps is set to {dataset.fps}, not using default')
                        setattr(model, 'fps', dataset.fps)
                elif getattr(model, 'nframe', 0) == 0:
                    raise ValueError(f'nframe is not suitable for {model_name}')
                else:
                    setattr(model, 'fps', None)
            if (
                'Qwen2-VL' in model_name
                or 'Qwen2.5-VL' in model_name
                or 'Qwen2.5-Omni' in model_name
            ):
                if getattr(model, 'nframe', None) is None and dataset.nframe > 0:
                    print(f'using {model_name} default setting for video, dataset.nframe is ommitted')
                if getattr(model, 'fps', None) is None and dataset.fps > 0:
                    print(f'using {model_name} default setting for video, dataset.fps is ommitted')

            if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name) and NOT_USE_SIBENCH_PROMPT:
                if dataset.nframe == 0:
                    raise ValueError(f'nframe must be set for custom prompt, fps is not suitable for {model_name}')
                struct = model.build_prompt(
                    dataset.data.iloc[i], dataset=dataset, video_llm=getattr(model, 'VIDEO_LLM', False)
                )
            else:
                struct = dataset.build_prompt(
                    dataset.data.iloc[i], video_llm=getattr(model, 'VIDEO_LLM', False), data_base=data_base
                )

            # If `SKIP_ERR` flag is set, the model will skip the generation if error is encountered
            if os.environ.get('SKIP_ERR', False) == '1':
                FAIL_MSG = 'Failed to obtain answer'
                try:
                    response = model.generate(message=struct, dataset=dataset_name)
                except RuntimeError as err:
                    torch.cuda.synchronize()
                    warnings.error(f'{type(err)} {str(err)}')
                    response = f'{FAIL_MSG}: {type(err)} {str(err)}'
            else:
                response = model.generate(message=struct, dataset=dataset_name)
        else:
            torch.cuda.empty_cache()
            raise NotImplementedError
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
def infer_data_job_mixed(
    model, work_dir, model_name, dataset, actual_dataset_name, verbose=False, api_nproc=4, ignore_failed=False, use_vllm=False
):
    lmu_path = LMUDataRoot()
    data_base = lmu_path
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    result_file = osp.join(work_dir, f'{model_name}_{actual_dataset_name}.xlsx')

    prev_file = f'{work_dir}/{model_name}_{actual_dataset_name}_PREV.pkl'
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{actual_dataset_name}.pkl')
    out_file = tmpl.format(rank)

    model = infer_data(
        model=model, work_dir=work_dir, model_name=model_name, dataset=dataset, actual_dataset_name=actual_dataset_name,
        data_base=data_base, out_file=out_file, verbose=verbose, api_nproc=api_nproc, use_vllm=use_vllm)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        data['prediction'] = [str(data_all[x]) for x in data['index']]
        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    if world_size > 1:
        dist.barrier()
    return model
