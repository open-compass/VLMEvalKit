import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
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
def infer_data_api(work_dir, model_name, dataset, nframe=8, pack=False, samples_dict={}, api_nproc=4, fps=-1):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    model = supported_VLM[model_name]() if isinstance(model_name, str) else model_name
    assert getattr(model, 'is_api', False)

    indices = list(samples_dict.keys())
    structs = [dataset.build_prompt(samples_dict[idx], num_frames=nframe,
                                    video_llm=getattr(model, 'VIDEO_LLM', False), fps=fps) for idx in indices]

    packstr = 'pack' if pack else 'nopack'
    if nframe > 0:
        out_file = f'{work_dir}/{model_name}_{dataset_name}_{nframe}frame_{packstr}_supp.pkl'
    else:
        out_file = f'{work_dir}/{model_name}_{dataset_name}_{fps}fps_{packstr}_supp.pkl'
    res = load(out_file) if osp.exists(out_file) else {}

    structs = [s for i, s in zip(indices, structs) if i not in res or res[i] == FAIL_MSG]
    indices = [i for i in indices if i not in res or res[i] == FAIL_MSG]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    return res


def infer_data(model_name, work_dir, dataset, out_file, nframe=8, pack=False, verbose=False, api_nproc=4, fps=-1):
    res = load(out_file) if osp.exists(out_file) else {}
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name

    sample_indices = list(dataset.videos) if pack else list(dataset.data['index'])
    samples = list(dataset.videos) if pack else list(range(len(dataset.data)))
    sample_map = {i: s for i, s in zip(sample_indices, samples)}

    sample_indices_sub = sample_indices[rank::world_size]
    if np.all([idx in res for idx in sample_indices_sub]):
        return model_name
    sample_indices_subrem = [x for x in sample_indices_sub if x not in res]

    model = supported_VLM[model_name]() if isinstance(model_name, str) else model_name

    is_api = getattr(model, 'is_api', False)
    if is_api:
        assert world_size == 1
        supp = infer_data_api(
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            nframe=nframe,
            pack=pack,
            samples_dict={k: sample_map[k] for k in sample_indices_subrem},
            api_nproc=api_nproc,
            fps=fps)
        for k in sample_indices_subrem:
            assert k in supp
        res.update(supp)
        dump(res, out_file)
        return model_name

    for i, idx in tqdm(enumerate(sample_indices_subrem)):
        if idx in res:
            continue
        if getattr(model, 'nframe', 0) > 0:
            if nframe > 0:
                if getattr(model, 'nframe', 0) != nframe:
                    print(f'{model_name} is a video-llm model, nframe is set to {nframe}, not using default')
                    setattr(model, 'nframe', nframe)
            elif getattr(model, 'fps', 0) == 0:
                raise ValueError(f'fps is not suitable for {model_name}')
        if getattr(model, 'fps', 0) > 0:
            if fps > 0:
                if getattr(model, 'fps', 0) != fps:
                    print(f'{model_name} is a video-llm model, fps is set to {fps}, not using default')
                    setattr(model, 'fps', fps)
            elif getattr(model, 'nframe', 0) == 0:
                raise ValueError(f'nframe is not suitable for {model_name}')
        if 'SUB_DATASET' in dataset.data.iloc[sample_map[idx]]:
            dataset_name = dataset.data.iloc[sample_map[idx]]['SUB_DATASET']
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            if nframe == 0:
                raise ValueError(f'nframe must be set for custom prompt, fps is not suitable for {model_name}')
            struct = model.build_prompt(
                dataset.data.iloc[sample_map[idx]], dataset=dataset,
                num_frames=nframe, video_llm=getattr(model, 'VIDEO_LLM', False)
            )
        else:
            struct = dataset.build_prompt(
                sample_map[idx], num_frames=nframe,
                video_llm=getattr(model, 'VIDEO_LLM', False), fps=fps
            )
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
        dataset,
        nframe=8,
        pack=False,
        verbose=False,
        subtitle=False,
        api_nproc=4,
        fps=-1):

    dataset_name = dataset.dataset_name
    packstr = 'pack' if pack else 'nopack'
    rank, world_size = get_rank_and_world_size()
    if nframe > 0:
        result_file = osp.join(work_dir, f'{model_name}_{dataset_name}_{nframe}frame_{packstr}.xlsx')
    else:
        result_file = osp.join(work_dir, f'{model_name}_{dataset_name}_{fps}fps_{packstr}.xlsx')
    if dataset_name == 'Video-MME':
        subtitle_str = 'subs' if subtitle else 'nosubs'
        result_file = result_file.replace('.xlsx', f'_{subtitle_str}.xlsx')
    if fps > 0:
        result_file = result_file.replace('.xlsx', f'_fps{fps}.xlsx')
    # Dump Predictions to Prev File if result file exists
    if osp.exists(result_file):
        return model_name

    if nframe > 0:
        tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}_{nframe}frame_{packstr}.pkl')
    else:
        tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}_{fps}fps_{packstr}.pkl')
    if dataset_name == 'Video-MME':
        subtitle_str = 'subs' if subtitle else 'nosubs'
        tmpl = tmpl.replace('.pkl', f'_{subtitle_str}.pkl')

    if fps > 0:
        tmpl = tmpl.replace('.pkl', f'_fps{fps}.pkl')
    out_file = tmpl.format(rank)

    model = infer_data(
        model,
        work_dir=work_dir,
        dataset=dataset,
        nframe=nframe,
        pack=pack,
        out_file=out_file,
        verbose=verbose,
        api_nproc=api_nproc,
        fps=fps)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        meta = dataset.data
        if dataset_name == 'MMBench-Video' and pack:
            meta, vstats = dataset.load_pack_answers(data_all)
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
