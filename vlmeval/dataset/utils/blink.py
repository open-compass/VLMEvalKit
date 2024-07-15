from ...smp import *
from ...utils import can_infer

FAIL_MSG = 'Failed to obtain answer via API.'


def build_prompt(question, options, prediction):
    tmpl = (
        "You are an AI assistant who will help me to match an answer with several options of a single-choice question. "
        "You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer. "
        "If the answer says things like refuse to answer, I'm sorry cannot help, etc., output (Z)"
        "If the meaning of all options are significantly different from the answer, or the answer does not select any option, output (Z)"\
        "Your should output one of the choices, (A),(B),(C),(D),(E) (if they are valid options), or (Z)\n"
        "Example 1: \n"
        "Question: Which point is closer to the camera?\nSelect from the following choices.\nOptions: (A) Point A\n(B) Point B\n(Z) Failed\nAnswer: Point B, where the child is sitting, is closer to the camera.\nYour output: (B)\n"
        "Example 2: \n"
        "Question: Which point is closer to the camera?\nSelect from the following choices.\nOptions: (A) Point A\n(B) Point B\n(Z) Failed\nAnswer: I'm sorry, but I can't assist with that request.\nYour output: (Z)\n"
        "Example 3: \n"
        "Question: Which point is corresponding to the reference point?\nSelect from the following choices.\nOptions: (A) Point A\n(B) Point B\n(Z) Failed\nAnswer:The reference point (REF) on the first image is at the tip of the pot, which is the part used to Poke if the pots were used for that action. Looking at the second image, we need to find the part of the object that would correspond to poking.\n(A) Point A is at the tip of the spoon's handle, which is not used for poking.\n(B) Point B is at the bottom of the spoon, which is not used for poking.\n(C) Point C is on the side of the pspoonot, which is not used for poking.\n(D) Point D is at the tip of the spoon, which is not used for poking.\n\nTherefore, there is no correct answer in the choices\nYour output: (Z)\n"
        "Example 4: \n"
        "Question: {}?\nOptions: {}\n(Z) Failed\nAnswer: {}\nYour output: "
    )
    return tmpl.format(question, options, prediction)


def build_blink_gpt4_prompt(line):

    question = line['question']
    prediction = str(line['prediction'])
    options = line['choices']
    
    prompt = build_prompt(question, options, prediction)
    return prompt


def BLINK_auxeval(model, line):
    prompt = build_blink_gpt4_prompt(line)
    log = ''
    retry = 5
    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse choice.\n'
        else:
            log += 'Succeed'
            return dict(log=log, res=res)
    log += 'All 5 retries failed to get multiple choice.\n'
    return dict(log=log, res='')


def BLINK_acc(result_file):
    data = load(result_file)
    tot = defaultdict(lambda: 0)
    fetch = defaultdict(lambda: 0)
    hit = defaultdict(lambda: 0)
    lt = len(data)
    
    for i in range(lt):
        item = data.iloc[i]
        print('item', item)
        cate = item['task']
        tot['Overall'] += 1
        tot[cate] += 1
        if item['log'] == 'Succeed':
            fetch['Overall'] += 1
            fetch[cate] += 1
        if item['res'] == item['answer']:
            hit['Overall'] += 1
            hit[cate] += 1
    res = defaultdict(list)
    for k in tot.keys():
        res['Tasks'].append(k)
        res['tot'].append(tot[k])
        res['prefetch'].append(fetch[k])
        res['hit'].append(hit[k])
        res['prefetch_rate'].append(fetch[k] / tot[k] * 100)
        res['acc'].append(hit[k] / tot[k] * 100)
    res = pd.DataFrame(res)
    return res
