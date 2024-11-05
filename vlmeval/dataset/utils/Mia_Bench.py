




def generate_prompt(d, response):
    instruction = d['question']
    weight = d['component_weight'] * 1
    weight = [int(i) for i in weight[1:-1].split(', ')]

    #d['num_of_component'] = len(d['components'])
    for i in range(len(weight)):
        weight[i] = str(weight[i])
    if d['num_of_component'] == 1:
        components = '''The first component is:' ''' + d['components'][0] + "'"
        score = '''The first component is worth ''' + weight[0] + ' scores.'
    elif d['num_of_component'] == 2:
        components = '''The first component is:' ''' + d['components'][0] + '''', and the second component is:' ''' + d['components'][1] + "'"
        score = '''The first and second component is each worth ''' + weight[0] + ' and ' + weight[1]+ ' scores.'
    elif d['num_of_component'] == 3:
        components = '''The first component is:' ''' + d['components'][0] + '''', and the second component is:' ''' + d['components'][1] + '''', and the third component is:' ''' + d['components'][2] + "'"
        score = '''The first second, and third component is each worth ''' + weight[0] + ', ' + weight[1]+ ' and ' + weight[2] + ' scores.'
    elif d['num_of_component'] == 4:
        components = '''The first component is:' ''' + d['components'][0] + '''', and the second component is:' ''' + d['components'][1] + '''', and the third component is:' ''' + d['components'][2] +  '''', and the fourth component is:' ''' + d['components'][3] + "'"
        score = '''The first second, third, and fourth component is each worth ''' + weight[0] + ', ' + weight[1]+ ', ' + weight[2] + ' and ' + weight[3] + ' scores.'
    elif d['num_of_component'] == 5:
        components = '''The first component is:' ''' + d['components'][0] + '''', and the second component is:' ''' + d['components'][1] + '''', and the third component is:' ''' + d['components'][2] +  '''', and the fourth component is:' ''' + d['components'][3] +  '''', and the fifth component is:' ''' + d['components'][4] + "'"
        score = '''The first second, third, fourth and fifth component is each worth ''' + weight[0] + ', ' + weight[1]+ ', ' + weight[2] + ', ' + weight[3] + ' and ' + weight[4] + ' scores.'
    return '''Here is an instruction for a multimodal LLM: ' ''' + instruction + ''' You need to grade if the response from the model follows each component of the instruction. ''' + components + ''' The response is:' '''  + response +  '''' You need to score the response and be strict. The total score ranges from 0 to 10, depending on if the response follows the instruction. ''' + score + ' List scores of each component, and the total score in one sentence in this format: score of component 1: x/2, score of component 2: y/8, total score: z/10. Then explain your reasons.'


def process_rawscore(component_type, raw_score):

    first_sentence = raw_score.split('''.''')[0].split(''',''')
    score_dict = {}
    for i in range(len(first_sentence) - 1):
        score_ = first_sentence[i].split(''':''')[1][1:].split('''/''')
        score = int(score_[0])/int(score_[1])
        score_dict[component_type[i]] = score
    total_score_ = first_sentence[i+1].split(''':''')[1][1:].split('''/''')
    total_score = int(total_score_[0])/int(total_score_[1])
    score_dict['total_score'] = total_score
    return score_dict

def get_score_dict(data, score_raw):
    cat_score_dict = {}
    for i in range(len(data)):
        try:
            cmp = data['component_type'][i][2:-2]
            cmp_list = cmp.split('\', \'')
            score_dict = process_rawscore(cmp_list, score_raw[i])
            for key, val in score_dict.items():
                if key not in cat_score_dict.keys():
                    cat_score_dict[key] = [val]
                else:
                    cat_score_dict[key].append(val)
        except:
            pass
    cat_score_dict_average = {}
    for key, val in cat_score_dict.items():
        cat_score_dict_average[key] = sum(val)/len(val)
    return cat_score_dict_average


