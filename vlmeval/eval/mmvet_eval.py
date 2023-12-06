from vlmeval.chat_api import OpenAIWrapper, OpenAIWrapperInternal
from vlmeval.smp import *
from vlmeval.utils import track_progress_rich
from collections import defaultdict

def build_mmvet_gpt4_prompt(line):
    question = line['question']
    gt = str(line['answer'])
    prediction = str(line['prediction'])
    prompt = """Compare the ground truth and prediction from AI models, to give a correctness score for the prediction. <AND> in the ground truth means it is totally right only when all elements in the ground truth are present in the prediction, and <OR> means it is totally right when any one element in the ground truth is present in the prediction. The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Just complete the last space of the correctness score.

    Question | Ground truth | Prediction | Correctness
    --- | --- | --- | ---
    What is x in the equation? | -1 <AND> -5 | x = 3 | 0.0
    What is x in the equation? | -1 <AND> -5 | x = -1 | 0.5
    What is x in the equation? | -1 <AND> -5 | x = -5 | 0.5
    What is x in the equation? | -1 <AND> -5 | x = -5 or 5 | 0.5
    What is x in the equation? | -1 <AND> -5 | x = -1 or x = -5 | 1.0
    Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme talks about Iceland and Greenland. It's pointing out that despite their names, Iceland is not very icy and Greenland isn't very green. | 0.4
    Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme is using humor to point out the misleading nature of Iceland's and Greenland's names. Iceland, despite its name, has lush green landscapes while Greenland is mostly covered in ice and snow. The text 'This is why I have trust issues' is a playful way to suggest that these contradictions can lead to distrust or confusion. The humor in this meme is derived from the unexpected contrast between the names of the countries and their actual physical characteristics. | 1.0
    """
    gpt4_prompt = prompt + '\n' + ' | '.join([question, gt.replace("<AND>", " <AND> ").replace("<OR>", " <OR> "), prediction, ""]) + "\nPredict the correctness of the answer (digit): "
    return gpt4_prompt

def MMVet_auxeval(model, line):
    prompt = build_mmvet_gpt4_prompt(line)
    output = model.generate(prompt)
    return output


def MMVet_acc(result_file):
    data = load(result_file)
    tot = defaultdict(lambda: 0)
    score = defaultdict(lambda: 0)
    lt = len(data)
    for i in range(lt):
        item = data.iloc[i]
        cate = item['category']
        grade = float(item['score'])
        cate_list = ['rec','ocr','know','gen','spat','math']
        for capa in cate_list:
            if capa in cate:
                tot[capa] += 1
                score[capa] += grade
        tot['Overall'] += 1
        score['Overall'] += grade

    res = defaultdict(list)
    for k in tot.keys():
        res['Category'].append(k)
        res['tot'].append(tot[k])
        res['acc'].append(score[k] / tot[k] * 100)
    res = pd.DataFrame(res)
    result_file = result_file.replace('.xlsx','_score.xlsx')
    dump(res,result_file)
    return res

def MMVet_eval(args):
    data = load(args.data)
    gpt_version = "gpt-4-0613"
    gpt_model = OpenAIWrapperInternal(model= gpt_version, max_tokens=3)
    storage = args.data.replace('.xlsx', '_gpt4.xlsx')
    lt = len(data)
    lines = [data.iloc[i] for i in range(lt)]

    if osp.exists(storage):
        data = load(storage)
        failed = data[data['score'] == 'Failed to obtain answer via API.']
        indices = list(failed['index'])
        score_map = data['score']
        for k in tqdm(indices): 
            score_map[k-1] = MMVet_auxeval(gpt_model, lines[k-1])
        data['score'] = [score_map[idx-1] for idx in data['index']]
    else:
        indices = list(data['index'])

        score_map = defaultdict(lambda:0.0)
        for k in tqdm(indices): 
            score_map[k] = MMVet_auxeval(gpt_model, lines[k-1])
            print(score_map[k])
        data['score'] = [score_map[idx] for idx in data['index']]
    dump(data, storage)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference LLM Answers. ")
    parser.add_argument("--data", type=str, help="The question set for inference, in excel / tsv / json format. ")
    #parser.add_argument("--model", type=str, help="The LLM (GPT) used for inference. ", default="gpt-3.5-turbo-0613", choices=['gpt-3.5-turbo-0613'])
    parser.add_argument("--nproc", type=int, default=4)
    #parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    MMVet_eval(args)
    storage = args.data.replace('.xlsx', '_gpt4.xlsx')
    MMVet_acc(storage)