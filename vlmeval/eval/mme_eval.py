from vlmeval.chat_api import OpenAIWrapper, OpenAIWrapperInternal
from vlmeval.smp import *
from vlmeval.utils import track_progress_rich

INTERNAL = os.environ.get('INTERNAL', 0)

def MME_rating(data_file):
    data = load(data_file)
    stats = defaultdict(dict)
    lt = len(data)
    for i in range(lt):
        item = data.iloc[i]
        category = item['category']
        image_path = item['image_path']
        score = item['score']
        if image_path not in stats[category]:
            stats[category][image_path] = []
        stats[category][image_path].append(score)
    
    def acc(key, mode='normal'):
        res = stats[key]
        values = []
        for val in res.values():
            if mode == 'normal':
                values.extend(val)
            elif mode == 'plus':
                values.append(val[0] * val[1])
        return np.mean(values) * 100
            
    scores = {}
    for k in stats:
        scores[k] = acc(k) + acc(k, 'plus')

    super_cates = dict(
        perception=['OCR', 'artwork', 'celebrity', 'color', 'count', 'existence', 'landmark', 'position', 'posters', 'scene'],
        reasoning=['code_reasoning', 'commonsense_reasoning', 'numerical_calculation', 'text_translation']
    )
    
    ret = {}
    for sc, cate_list in super_cates.items():
        base = 0
        for c in cate_list:
            base += scores[c]
        ret[sc] = base 
    ret.update(scores)
    ret = d2df(ret)
    return ret

def MME_postproc(data):
    data['yes'] = data["prediction"].str.contains("Yes", case=False)
    data["no"] = data["prediction"].str.contains("No", case=False)
    data['raw_prediction'] = data['prediction']
    data['prediction'] = data.apply(
        lambda x: "Yes" if x["yes"] and not x["no"] else "No" if x["no"] and not x["yes"] else "Unknown", axis=1
    )
    data.drop(["yes", "no"], axis=1, inplace=True)
    data["score"] = (data["answer"] == data["prediction"])
    return data

def MME_build_matching_prompt(line):
    tmpl = (
        "You are an AI assistant who will help me to match an answer with two options of a question. "
        "The options are only Yes / No. "
        "You are provided with a question and an answer, and you need to find which option (Yes / No) is most similar to the answer. "
        "If the meaning of all options are significantly different from the answer, output Unknown. "\
        "Your should output a single word among the following 3 choices: Yes, No, Unknown.\n"
        "Example 1: \n"
        "Question: Is the word in this image 'Hello'?\nAnswer: The word in this image is 'Hello'.\nYour output: Yes\n"
        "Example 2: \n"
        "Question: Is the word in this image 'Hello'?\nAnswer: The word in this image is not 'Hello'.\nYour output: No\n"
        "Example 3: \n"
        "Question: {}?\nAnswer: {}\nYour output: "
    )
    return tmpl.format(line['question'], line['raw_prediction'])

def MME_answer_extraction(output):
    s = output.lower()
    if 'yes' in s and 'no' not in s:
        return 'Yes'
    if 'yes' not in s and 'no' in s:
        return 'No'
    return 'Unknown'

def MME_auxeval(model, line):
    prompt = MME_build_matching_prompt(line)
    output = model.generate(prompt)
    ans = MME_answer_extraction(output)
    return ans

def MME_auxeval_tup(tup):
    model, line = tup
    return MME_auxeval(model, line)

def MME_eval(eval_file, model='chatgpt-0613', nproc=4, verbose=False):
    logger = get_logger('Evaluation')
    
    data = load(eval_file)
    if 'raw_prediction' not in data:
        data = MME_postproc(data)

    preds_map = {x: y for x, y in zip(data['index'], data['prediction'])}
    unknown = data[data['prediction'] == 'Unknown']
    storage = eval_file.replace('.xlsx', '_auxmatch.xlsx')
    
    if not osp.exists(storage):
        assert model == 'chatgpt-0613'
        model_name = 'gpt-3.5-turbo-0613'

        if INTERNAL:
            model = OpenAIWrapperInternal(model_name, verbose=verbose)
        else:
            model = OpenAIWrapper(model_name, verbose=verbose)

        lt = len(unknown)
        lines = [unknown.iloc[i: i + 1] for i in range(lt)]
        tups = [(model, line) for line in lines]
        indices = list(unknown['index'])

        if len(tups):
            # Do not save temporary file due to the fast speed
            res = track_progress_rich(MME_auxeval, tups, nproc=nproc, chunksize=nproc)

            for k, v in zip(indices, res):
                preds_map[k] = v

        data['prediction'] = [preds_map[idx] for idx in data['index']]
        dump(data, storage)
    else:
        logger.warning(f"GPT matching file {storage} already exists, will reuse it in MME_eval. ")
    
    data = load(storage)
    data["score"] = (data["answer"] == data["prediction"])
    dump(data, storage)
    score = MME_rating(storage)
    score_tgt = storage.replace('auxmatch.xlsx', 'score.csv')
    dump(score, score_tgt)
    logger.info(f'MME_eval successfully finished evaluating {eval_file}, results saved in {score_tgt}')
    logger.info('Score: ')
    logger.info(score)
    return score

def parse_args():
    parser = argparse.ArgumentParser(description="Inference LLM Answers. ")
    parser.add_argument("data", type=str, help="The question set for inference, in excel / tsv / json format. ")
    parser.add_argument("--model", type=str, help="The LLM (GPT) used for inference. ", default="chatgpt-0613", choices=['chatgpt-0613'])
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    acc = MME_eval(eval_file=args.data, model=args.model, nproc=args.nproc, verbose=args.verbose)
