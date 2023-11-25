import torch 
import torch.distributed as dist
from vlmeval.mllm.models import model_cls_map
from vlmeval import *

def build_mmbench_prompt(img_dir, line):
    os.makedirs(img_dir, exist_ok=True)
    idx = line['index']
    img = line['image']
    tgt_path = osp.join(img_dir, f'{idx}.jpg')
    decode_base64_to_image_file(img, tgt_path)

    question = line['question']
    option_candidate = ['A', 'B', 'C', 'D', 'E']
    options = {
        cand: line[cand]
        for cand in option_candidate
        if cand in line and not pd.isna(line[cand])
    }
    options_prompt = 'Options:\n'
    for key, item in options.items():
        options_prompt += f'{key}. {item}\n'

    hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
    prompt = ''
    if hint is not None:
        prompt += f'Hint: {hint}\n'
    prompt += f'Question: {question}\n'
    prompt += options_prompt
    prompt += 'Please select the correct answer from the options above. \n'
    return {'image': tgt_path, 'text': prompt}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument("--model", type=str, nargs='+', required=True)
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()
    return args

def infer_data(model_name, data, out_file, verbose=False, IMG_DIR='mmbench_images'):
    res = {}
    if osp.exists(out_file):
        res = load(out_file)
    lt = len(data)

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        return 

    if isinstance(model_name, str):
        model = model_cls_map[model_name]()
    else:
        model = model_name

    for i in tqdm(range(lt)):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        if hasattr(model, 'build_mmbench_prompt'):
            struct = model.build_mmbench_prompt(IMG_DIR, data.iloc[i])
        else:
            struct = build_mmbench_prompt(IMG_DIR, data.iloc[i])

        if hasattr(model, 'mmbench_generate'):
            response = model.mmbench_generate(prompt=struct['text'], image_path=struct['image'])    
        else:
            response = model.generate(prompt=struct['text'], image_path=struct['image'])

        torch.cuda.empty_cache()
        
        if verbose:
            print(response, flush=True)
        res[idx] = response
        if (i + 1) % 20 == 0:
            dump(res, out_file)
    dump(res, out_file)
    return model

def prefetch_acc(result_file):
    data = load(result_file)
    from vlmeval.mllm.llm_fastroll_0630 import build_choices, can_infer
    tot = defaultdict(lambda: 0)
    match = defaultdict(lambda: 0)
    hit = defaultdict(lambda: 0)
    lt = len(data)
    for i in range(lt):
        item = data.iloc[i]
        cate = item['category']
        tot['Overall'] += 1
        tot[cate] += 1
        choices = build_choices(item)
        matched = can_infer(item['prediction'], choices)
        if matched:
            match['Overall'] += 1
            match[cate] += 1
            if matched == item['answer']:
                hit['Overall'] += 1
                hit[cate] += 1
    res = defaultdict(list)
    for k in tot.keys():
        res['Category'].append(k)
        res['tot'].append(tot[k])
        res['match'].append(match[k])
        res['hit'].append(hit[k])
        res['match_rate'].append(match[k] / tot[k] * 100)
        res['acc'].append(hit[k] / match[k] * 100)
    res = pd.DataFrame(res)
    dump(res, result_file.replace('.xlsx', '_score.xlsx'))
    return res
        
def main():
    args = parse_args()
    assert len(args.data), "--data should be a list of data files"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')

    for _, model_name in enumerate(args.model):
        model = None
        for i, data_file in enumerate(args.data):
            IMG_DIR = 'mmbench_images'
            if 'ccbench' in data_file.lower():
                IMG_DIR = 'ccbench_images'
            elif 'seed' in data_file.lower():
                IMG_DIR = 'seedbench_images'

            data = load(data_file)
            data_base_name = osp.basename(data_file)
            
            tmpl = data_file.replace(data_base_name, f'{model_name}_' + '{}' + f'{world_size}_' + data_base_name)
            tmpl = tmpl.replace('.tsv', '.pkl')

            data = data.iloc[local_rank::world_size]
            out_file = tmpl.format(local_rank)

            result_file = data_file.replace(data_base_name, f'{model_name}_' + data_base_name).replace('.tsv', '.xlsx')
            
            if model is None:
                model = model_name # which is only a name

            if not osp.exists(result_file):
                model = infer_data(model, data, out_file, verbose=args.verbose, IMG_DIR=IMG_DIR)
                if world_size > 1:
                    dist.barrier()

                if local_rank == 0:
                    data_all = {}
                    for i in range(world_size):
                        data_all.update(load(tmpl.format(i)))

                    data = load(data_file)
                    assert len(data_all) == len(data)
                    
                    data['prediction'] = [data_all[x] for x in data['index']]
                    data.pop('image')
                    dump(data, result_file)   
                    for i in range(world_size):
                        os.remove(tmpl.format(i))
                         
            if local_rank == 0:
                time.sleep(3)
                res = prefetch_acc(result_file)
                print(model_name, res)


if __name__ == '__main__':
    main()