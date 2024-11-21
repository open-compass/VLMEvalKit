import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *
from vlmeval.api.prism_llm import Reasoning

import os


FAIL_MSG = 'Failed to obtain answer via API.'


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data', type=str, nargs='+', required=True)
#     parser.add_argument('--model', type=str, nargs='+', required=True)
#     parser.add_argument('--nproc', type=int, default=4, required=True)
#     parser.add_argument('--verbose', action='store_true')
#     args = parser.parse_args()
#     return args


# Only API model is accepted
def prism_infer_data_api(work_dir, model_name, dataset, index_set=None, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model_name, str) else model_name
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    # structs = [dataset.build_prompt(data.iloc[i]) for i in range(lt)]

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'
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





def prism_infer_data_fronted(model_name_fronted, model_name_backend, work_dir, dataset, out_file, verbose=False, api_nproc=4):
    dataset_name = dataset.dataset_name
    prev_file_fronted = f'{work_dir}/Prism_fronted_{model_name_fronted}_{model_name_backend}_{dataset_name}_PREV.pkl'
    res = load(prev_file_fronted) if osp.exists(prev_file_fronted) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return

    data = data[~data['index'].isin(res)]
    lt = len(data)

    model = supported_VLM[model_name_fronted]() if isinstance(model_name_fronted, str) else model_name_fronted

    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp = prism_infer_data_api(
            work_dir=work_dir,
            model_name=model_name_fronted,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model_name_fronted
    else:
        model.set_dump_image(dataset.dump_image)

    for i in tqdm(range(lt)):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(data.iloc[i])

        response = model.generate(message=struct, dataset=dataset_name)
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 10 == 0:
            dump(res, out_file)

    res = {k: res[k] for k in data_indices}
    dump(res, out_file)

    return model


def prism_infer_data_job_fronted(model_fronted, work_dir, model_name_fronted, model_name_backend, dataset, verbose=False, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name

    result_file_fronted = osp.join(work_dir, f'Prism_fronted_{model_name_fronted}_{model_name_backend}_{dataset_name}.xlsx')
    prev_file_fronted = f'{work_dir}/Prism_fronted_{model_name_fronted}_{model_name_backend}_{dataset_name}_PREV.pkl'

    if osp.exists(result_file_fronted):
        if rank == 0:
            data = load(result_file_fronted)
            results = {k: v for k, v in zip(data['index'], data['fronted_description'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file_fronted)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, 'Prism_fronted_' + '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)

    model_fronted = prism_infer_data_fronted(
        model_name_fronted, model_name_backend, work_dir=work_dir, dataset=dataset, out_file=out_file, verbose=verbose, api_nproc=api_nproc)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        data['fronted_description'] = [str(data_all[x]) for x in data['index']]
        if 'image' in data:
            data.pop('image')

        dump(data, result_file_fronted)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    if world_size > 1:
        dist.barrier()

    return model_fronted


def prism_infer_data_backend(model_name_fronted, model_name_backend, work_dir, dataset, out_file, verbose=False, api_nproc=4):
    dataset_name = dataset.dataset_name
    prev_file_backend = f'{work_dir}/Prism_backend_{model_name_fronted}_{model_name_backend}_{dataset_name}_PREV.pkl'
    res = load(prev_file_backend) if osp.exists(prev_file_backend) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return

    data = data[~data['index'].isin(res)]
    lt = len(data)

    breakpoint()
    model = Reasoning(model=model_name_backend)

    for i in tqdm(range(lt)):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        des = data.iloc[i]['fronted_description']
        text = data.iloc[i]['question']
        response = model.generate(des, text)

        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response

        if (i + 1) % 10 == 0:
            dump(res, out_file)

    res = {k: res[k] for k in data_indices}
    dump(res, out_file)

    return model


    # model = supported_VLM[model_name_backend]() if isinstance(model_name_backend, str) else model_name_backend
    #
    # is_api = getattr(model, 'is_api', False)
    # if is_api:
    #     lt, indices = len(data), list(data['index'])
    #     supp = prism_infer_data_api(
    #         work_dir=work_dir,
    #         model_name=model_name_fronted,
    #         dataset=dataset,
    #         index_set=set(indices),
    #         api_nproc=api_nproc)
    #     for idx in indices:
    #         assert idx in supp
    #     res.update(supp)
    #     res = {k: res[k] for k in data_indices}
    #     dump(res, out_file)
    #     return model_name_fronted
    # else:
    #     model.set_dump_image(dataset.dump_image)
    #
    # for i in tqdm(range(lt)):
    #     idx = data.iloc[i]['index']
    #     if idx in res:
    #         continue
    #
    #     if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
    #         struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
    #     else:
    #         struct = dataset.build_prompt(data.iloc[i])
    #
    #     response = model.generate(message=struct, dataset=dataset_name)
    #     torch.cuda.empty_cache()
    #
    #     if verbose:
    #         print(response, flush=True)
    #
    #     res[idx] = response
    #     if (i + 1) % 10 == 0:
    #         dump(res, out_file)
    #
    # res = {k: res[k] for k in data_indices}
    # dump(res, out_file)
    #
    # return model


def prism_infer_data_job_backend(model_fronted, model_backend, work_dir, model_name_fronted, model_name_backend, dataset, verbose=False, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    result_file_fronted = osp.join(work_dir, f'Prism_fronted_{model_name_fronted}_{model_name_backend}_{dataset_name}.xlsx')
    dataset.data = load(result_file_fronted)

    result_file_backend = osp.join(work_dir, f'Prism_backend_{model_name_fronted}_{model_name_backend}_{dataset_name}.xlsx')
    prev_file_backend = f'{work_dir}/Prism_backend_{model_name_fronted}_{model_name_backend}_{dataset_name}_PREV.pkl'

    if osp.exists(result_file_backend):
        if rank == 0:
            data = load(result_file_backend)
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file_backend)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, 'Prism_backend_' + '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)

    model_backend = prism_infer_data_backend(
        model_fronted, model_backend, work_dir=work_dir, dataset=dataset, out_file=out_file, verbose=verbose, api_nproc=api_nproc)
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

        dump(data, result_file_backend)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    if world_size > 1:
        dist.barrier()

    return model_backend


def prism_infer_data_job(model_fronted, model_backend, work_dir, model_name_fronted, model_name_backend, dataset, verbose=False, api_nproc=4, ignore_failed=False):
    logger = logging.getLogger('prism')

    logger.info(f'--implementing fronted model {model_name_fronted} for describing--')
    model_fronted = prism_infer_data_job_fronted(model_fronted, work_dir, model_name_fronted, model_name_backend, dataset, verbose=False, api_nproc=4, ignore_failed=False)
    logger.info(f'--finished description from fronted model {model_name_fronted} and saved results--')

    logger.info(f'--implementing fronted model {model_name_backend} for describing--')
    model_backend = prism_infer_data_job_backend(model_fronted, model_backend, work_dir, model_name_fronted, model_name_backend, dataset, verbose=False, api_nproc=4, ignore_failed=False)
    logger.info(f'--finished description from fronted model {model_name_backend} and saved results--')

    breakpoint()
    return model_fronted, model_backend


