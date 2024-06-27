from vlmeval.smp import *


def AesBench_eval(eval_file):
    AesBench_score = {
        'AesP': 0,
        'AesA': 0,
        'AesE': 0,
    }

    logger = get_logger('Evaluation')

    data = load(eval_file)
    lt = len(data)
    lines = [data.iloc[i] for i in range(lt)]
    for i in tqdm(range(len(lines))):
        line = lines[i]

        # GT
        predict = str(line['prediction'])

        # MLLM answer
        answers = eval(line['answer'])

        category = line['category']
        if category == 'AesP':
            for j in range(len(answers)):
                answer = answers[j]
                predict = predict[j]
                if answer in predict:
                    AesBench_score[category] += 1

        elif category == 'AesA':
            for j in range(len(answers)):
                answer = answers[j]
                predict = predict[j]
                if answer in predict:
                    AesBench_score[category] += 1
        elif category == 'AesE':
            for j in range(len(answers)):
                answer = answers[j]
                predict = predict[j]
                if answer in predict:
                    AesBench_score[category] += 1

    final_score_dict = {}
    final_score_dict['AesP'] = final_score_dict['AesP'] / 40.0
    final_score_dict['AesA'] = final_score_dict['AesA'] / 40.0
    final_score_dict['AesE'] = final_score_dict['AesE'] / 40.0

    score_pth = eval_file.replace('.xlsx', '_score.json')
    dump(final_score_dict, score_pth)
    logger.info(f'AesBench(val) successfully finished evaluating {eval_file}, results saved in {score_pth}')
    logger.info('Score: ')
    for key, value in final_score_dict.items():
        logger.info('{}:{}'.format(key, value))
