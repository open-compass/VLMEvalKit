from vlmeval.smp import *

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