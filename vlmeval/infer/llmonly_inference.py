import torch 
import torch.distributed as dist
from vlmeval.chat_api.inference import model_map as llm_model_map
from vlmeval.utils.mp_util import track_progress_rich
from vlmeval.mllm.util import can_infer_option
from vlmeval import *

def build_options(option_list):
    chars = string.ascii_uppercase
    s = 'There are several options: \n'
    for c, opt in zip(chars, option_list):
        if not pd.isna(opt):
            s += f'{c}. {opt}\n'
        else:
            return s
    return s

prompt_tryurbest = """
You are an AI assistant which is designed to answer questions from people. 
You will be asked a question and provided several choices, and you need to choose the best choice to answer the question. 
There is an image associated with the problem, but that is not provided to you, so you need to try your best to hallucinate the image.
Again, you can only choose from the provided choices, and output a single uppercase letter (A, B, C, D, etc.). \n
Question: <question begins> {} <question ends>; \n
Choices: <choices begin> {} <choices end>. \n
Your answer: 
"""

prompt_responsive = """
You are an AI assistant which is designed to answer questions from people. 
You will be asked a question and provided several choices, and you need to choose the best choice to answer the question. 
There is an image associated with the problem, but that is not provided to you.
If it's impossible to answer the question without the image, just output a single uppercase letter E. 
Again, you can only choose from the provided choices, and output a single uppercase letter (A, B, C, D, E, etc.). \n
Question: <question begins> {} <question ends>; \n
Choices: <choices begin> {} <choices end>. \n
Your answer: 
"""

def build_prompt(line, mode='tryurbest'):
    tmpl = prompt_tryurbest if mode == 'tryurbest' else prompt_responsive
    question = line['question']
    options = [line[ch] for ch in 'ABCD']
    options = [x for x in options if not pd.isna(x)]
    option_str = build_options(options)

    prompt = tmpl.format(question, option_str)
    return prompt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt4")
    parser.add_argument("--mode", type=str, default="tryurbest")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()
    return args
        
def main():
    args = parse_args()
    assert len(args.data), "--data should be a list of data files"
    data = load(args.data)
    model = llm_model_map[args.model]()
    ans_map = {i: a for i, a in zip(data['index'], data['answer'])}

    out_path = f'{args.model}_{args.mode}.xlsx'
    tmp_path = f'{args.model}_{args.mode}.pkl'

    res, failed = {}, defaultdict(lambda: False)
    if osp.exists(tmp_path):
        res = load(tmp_path)

    locations = [int(1e6) * i for i in range(5)]
    st, ed = locations[:-1], locations[1:]

    for _ in range(4):
        group = data[st[_] <= data['index']]
        group = group[data['index'] < ed[_]]
        indices, prompts = [], []
        lt = len(group)
        for i in range(lt):
            line = group.iloc[i]
            idx = line['index']
            if failed[idx % 1000000]:
                res[idx] = 'X'
            else:
                indices.append(idx)
                prompts.append(build_prompt(line, args.mode))
        dump(res, tmp_path)
        results = track_progress_rich(
            model.generate, 
            prompts,
            keys=indices, 
            save=tmp_path, 
            nproc=args.nproc, 
            chunksize=args.nproc)
        res = load(tmp_path)
        for ind, s in zip(indices, results):
            if ind in res:
                assert res[ind] == s
            else:
                res[ind] = s
            infered = can_infer_option(s)
            if infered and (infered != ans_map[ind]):
                failed[ind % 1000000] = True
        dump(res, tmp_path)
    
    dump(res, tmp_path)
    tmp = load(tmp_path)
    data['prediction'] = [tmp[x] for x in data['index']]
    dump(data, out_path)

if __name__ == '__main__':
    main()