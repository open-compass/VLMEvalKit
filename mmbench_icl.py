from vlmeval.smp import *
from vlmeval.utils import TSVDataset
from functools import partial
import torch.distributed as dist
from tqdm import tqdm
from vlmeval.vlm import *
from vlmeval.api import *

dev = None
dataset_name = None

def dump_image(line, dataset):
    ROOT = LMUDataRoot()
    assert isinstance(dataset, str)
    img_root = osp.join(ROOT, 'images', dataset)
    os.makedirs(img_root, exist_ok=True)
    tgt_path = osp.join(img_root, f"{line['index']}.jpg")
    if not osp.exists(tgt_path):
        decode_base64_to_image_file(line['image'], tgt_path)
    return tgt_path

def build_prompt(line, database, nshot=1, specific=False):
    samples = database
    if specific:
        samples = samples[samples['category'] == line['category']]
        
    samples = samples.sample(frac=1, random_state=line['index']).reset_index(drop=True)
    icls = samples.iloc[:nshot].copy()

    prompt_list = []
    for i in range(nshot):
        sample = icls.iloc[i]
        tgt_path = dump_image(sample, 'mmbench_v11')

        options = {
            cand: sample[cand]
            for cand in string.ascii_uppercase
            if cand in sample and not pd.isna(sample[cand])
        }
        options_prompt = ''
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        question = sample['question']
        hint = sample['hint'] if ('hint' in sample and not pd.isna(sample['hint'])) else None
        prompt = f'Example {i + 1}:\n'
        if hint is not None:
            prompt += f'Hint:\n{hint}\n'
        prompt += f'Question:\n{question}\nOptions:\n{options_prompt}\nPlease select the correct answer from the options above. \n'
        prompt_list.append(prompt)
        prompt_list.append(tgt_path)
        prompt_list.append(f'Answer: {sample["answer"]}\n')
    
    tgt_path = dump_image(line, 'mmbench_v11')
    options = {
        cand: line[cand]
        for cand in string.ascii_uppercase
        if cand in line and not pd.isna(line[cand])
    }
    options_prompt = ''
    for key, item in options.items():
        options_prompt += f'{key}. {item}\n'
    question = line['question']
    hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
    prompt = f'Please solve the following problem:\n'
    if hint is not None:
        prompt += f'Hint:\n{hint}\n'
    prompt += f'Question:\n{question}\nOptions:\n{options_prompt}\nPlease select the correct answer from the options above. \n'
    prompt_list.append(prompt)
    prompt_list.append(tgt_path)
    prompt_list.append('Answer: ')
    return prompt_list
    
models = {
    'qwen_chat': partial(QwenVLChat, model_path='Qwen/Qwen-VL-Chat'),
    'idefics_9b_instruct': partial(IDEFICS, model_pth="HuggingFaceM4/idefics-9b-instruct"),
    'idefics_80b_instruct': partial(IDEFICS, model_pth="HuggingFaceM4/idefics-80b-instruct"),
    "mPLUG-Owl2": partial(mPLUG_Owl2, model_path='MAGAer13/mplug-owl2-llama2-7b'),
    'GPT4V': partial(GPT4V, model='gpt-4-vision-preview', temperature=0, img_size=512, img_detail='low', retry=10),
    'GeminiProVision': partial(GeminiProVision, temperature=0, retry=10),
    'QwenVLMax': partial(QwenVLAPI, model='qwen-vl-max', temperature=0, retry=10),
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='qwen_chat')
    parser.add_argument('--data', type=str, default='mmbench_v11')
    parser.add_argument("--nshot", type=int, default=1)
    parser.add_argument("--specific", action='store_true', help="Only use samples from the same category as the target sample")
    parser.add_argument("--nproc", type=int, default=4, help="Parallel API calling")
    parser.add_argument("--retry", type=int, default=None, help="retry numbers for API VLMs")
    args = parser.parse_args()
    return args

def infer_data_job(model, prompts, indices, out_name):
    res = {}
    if osp.exists(out_name):
        res = load(out_name)
    for i in tqdm(range(len(indices))):
        ind, prompt = indices[i], prompts[i]
        if ind in res:
            continue
        out = model.interleave_generate(prompt)
        res[ind] = out
        dump(res, out_name)
    return 

def main():
    args = parse_args()
    dataset_name = args.data
    dataset = TSVDataset(args.data)
    os.makedirs(args.model, exist_ok=True)

    data = dataset.data
    dev = data[data['split'] == 'dev'].copy()
    test = data[data['split'] == 'test'].copy()

    dev = dev[dev['index'] < 1e6]
    
    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=5400))
    
    assert args.model in models
    model = models[args.model]()
    
    prompts = [build_prompt(test.iloc[i], database=dev, nshot=args.nshot, specific=args.specific) for i in tqdm(range(len(test))) if i % world_size == rank]
    indices = list(test['index'])
    indices = indices[rank::world_size]
    
    spstr = 'specific' if args.specific else 'general'
    out_tmpl = args.model + '/{}' + f'{world_size}_{args.model}_{args.data}_{args.nshot}_{spstr}.pkl'
    out_file = out_tmpl.format(rank)
    result_file = f'{args.model}/{args.model}_{args.data}_{args.nshot}_{spstr}.xlsx'

    infer_data_job(model, prompts, indices, out_file)
    if world_size > 1: dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(out_tmpl.format(i)))

        test = dataset.data
        data = test[test['split'] == 'test'].copy()

        for x in data['index']:
            assert x in data_all
        data['prediction'] = [str(data_all[x]) for x in data['index']]
        data.pop('image')

        dump(data, result_file)             
        for i in range(world_size):
            os.remove(out_tmpl.format(i))
    
    from vlmeval.evaluate import multiple_choice_eval
    if rank == 0:
        multiple_choice_eval(result_file, dataset="default", model='gpt-4-0125', nproc=6, verbose=False)

if __name__ == '__main__':
    main()