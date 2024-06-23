import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.dataset import build_dataset, split_MMMU, MMBenchVideo
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *

FAIL_MSG = 'Failed to obtain answer via API.'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(work_dir, model_name, dataset_name, nframe=8, pack=False, samples_dict={}, api_nproc=4):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset = build_dataset(dataset_name, pack=pack)
    model = supported_VLM[model_name]() if isinstance(model_name, str) else model_name
    assert getattr(model, 'is_api', False)

    indices = list(samples_dict.keys())
    structs = [dataset.build_prompt(samples_dict[idx], nframe=nframe) for idx in indices]

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'
    res = {}
    if osp.exists(out_file):
        res = load(out_file)

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    # For now, we do not use split_MMMU for MMMU dataset
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    return res


def infer_data(model_name, work_dir, dataset_name, out_file, nframe=8, pack=False, verbose=False, api_nproc=4):
    res = load(out_file)
    rank, world_size = get_rank_and_world_size()
    if rank == 0:
        dataset = build_dataset(dataset_name, pack=pack)
    if world_size > 1:
        dist.barrier()
    dataset = build_dataset(dataset_name, pack=pack)

    sample_indices = list(dataset.videos) if pack else list(dataset.data['index'])
    samples = list(dataset.videos) if pack else list(range(len(dataset.data)))
    sample_map = {i: s for i, s in zip(sample_indices, samples)}

    sample_indices_sub = sample_indices[rank::world_size]
    if np.all([idx in res for idx in sample_indices_sub]):
        return model_name
    sample_indices_subrem = [x for x in sample_indices if x not in res]

    model = supported_VLM[model_name]() if isinstance(model_name, str) else model_name

    is_api = getattr(model, 'is_api', False)
    if is_api:
        assert world_size == 1
        supp = infer_data_api(
            work_dir=work_dir,
            model_name=model_name,
            dataset_name=dataset_name,
            nframe=nframe,
            pack=pack,
            samples_dict={k: sample_map[k] for k in sample_indices_subrem},
            api_nproc=api_nproc)
        for k in sample_indices_subrem:
            assert k in supp
        res.update(supp)
        dump(res, out_file)
        return model_name

    for i, idx in tqdm(enumerate(sample_indices_subrem)):
        if idx in res:
            continue
        struct = dataset.build_prompt(sample_map[idx], nframe=nframe)
        response = model.generate(message=struct, dataset=dataset_name)
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 20 == 0:
            dump(res, out_file)

    res = {k: res[k] for k in sample_indices_sub}
    dump(res, out_file)
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job_video(
        model,
        work_dir,
        model_name,
        dataset_name,
        nframe=8,
        pack=False,
        verbose=False,
        api_nproc=4):

    packstr = 'pack' if pack else 'nopack'
    rank, world_size = get_rank_and_world_size()
    result_file = osp.join(work_dir, f'{model_name}_{dataset_name}_{nframe}frame_{packstr}.xlsx')

    # Dump Predictions to Prev File if result file exists
    if osp.exists(result_file):
        return model_name

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}_{nframe}frame_{packstr}.pkl')
    out_file = tmpl.format(rank)

    model = infer_data(
        model,
        work_dir=work_dir,
        dataset_name=dataset_name,
        nframe=nframe,
        pack=pack,
        out_file=out_file,
        verbose=verbose,
        api_nproc=api_nproc)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        dataset = build_dataset(dataset_name, pack=pack)
        meta = dataset.data
        if dataset_name == 'MMBench-Video' and pack:
            meta, vstats = MMBenchVideo('MMBench-Video').load_pack_answers(data_all)
            print(f'Statitics of Pack Video Inference: {vstats}')
        else:
            for x in meta['index']:
                assert x in data_all
            meta['prediction'] = [str(data_all[x]) for x in meta['index']]
            if 'image' in meta:
                meta.pop('image')

        dump(meta, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    return model
