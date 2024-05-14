from vlmeval import *

inp = sys.argv[1] 
assert inp.endswith('.tsv')

data = load(inp)
OFFSET = 1e6
while max(data['index']) >= OFFSET:
    OFFSET *= 10

data_2c = data[pd.isna(data['C'])]
data_3c = data[~pd.isna(data['C']) & pd.isna(data['D'])]
data_4c = data[~pd.isna(data['D'])]
map_2c = [('AB', 'BA')]
map_3c = [('ABC', 'BCA'), ('ABC', 'CAB')]
map_4c = [('ABCD', 'BCDA'), ('ABCD', 'CDAB'), ('ABCD', 'DABC')]

def okn(o, n=4):
    ostr = o.replace(',', ' ')
    osplits = ostr.split()
    if sum([c in osplits for c in string.ascii_uppercase[:n - 1]]) == n - 1:
        return False
    olower = o.lower()
    olower = olower.replace(',', ' ')
    olower_splits = olower.split()
    if 'all' in olower_splits or 'none' in olower_splits:
        return False
    return True

yay4, nay4 = [], []
lt4 = len(data_4c)
for i in range(lt4):
    if okn(data_4c.iloc[i]['D'], 4):
        yay4.append(i)
    else:
        nay4.append(i)
data_4c_y = data_4c.iloc[yay4]
data_4c_n = data_4c.iloc[nay4]
data_3c = pd.concat([data_4c_n, data_3c])

yay3, nay3 = [], []
lt3 = len(data_3c)
for i in range(lt3):
    if okn(data_3c.iloc[i]['C'], 3):
        yay3.append(i)
    else:
        nay3.append(i)
data_3c_y = data_3c.iloc[yay3]
data_3c_n = data_3c.iloc[nay3]
data_2c = pd.concat([data_3c_n, data_2c])

def remap(data_in, tup, off):
    off = int(off)
    data = data_in.copy()
    char_map = {k: v for k, v in zip(*tup)}
    idx = data.pop('index')
    answer = data.pop('answer')
    answer_new = [char_map[x] if x in char_map else x for x in answer]
    data['answer'] = answer_new
    options = {}
    for c in char_map:
        options[char_map[c]] = data.pop(c)
    for c in options:
        data[c] = options[c]
    data.pop('image')
    data['image'] = idx
    idx = [x + off for x in idx]
    data['index'] = idx
    return data

data_all = pd.concat([
    data_2c, 
    data_3c_y, 
    data_4c_y, 
    remap(data_2c, map_2c[0], OFFSET),
    remap(data_3c_y, map_3c[0], OFFSET),
    remap(data_4c_y, map_4c[0], OFFSET),
    remap(data_3c_y, map_3c[1], OFFSET * 2),
    remap(data_4c_y, map_4c[1], OFFSET * 2),
    remap(data_4c_y, map_4c[2], OFFSET * 3),
])

tgt_file = inp.replace('.tsv', '_CIRC.tsv')
dump(data_all, tgt_file)
print(f'The circularized data is saved to {tgt_file}')
assert osp.exists(tgt_file)
print(f'The MD5 for the circularized data is {md5(tgt_file)}')
