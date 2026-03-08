import click
import string
import pandas as pd
from collections import defaultdict, deque
from vlmeval.smp import load, dump, md5
import warnings
import os.path as osp


@click.command()
@click.argument('input_file', type=str)
def CIRCULAR(input_file):
    def proc_str(s):
        chs = set(s)
        chs = [x for x in chs if x not in string.ascii_letters and x != ' ']
        for ch in chs:
            s = s.replace(ch, ' ')
        return s

    def abnormal_entry(line):
        choices = {k: line[k] for k in string.ascii_uppercase if k in line and not pd.isna(line[k])}
        has_label = False
        for k in choices:
            s = proc_str(choices[k]).split()
            hit_words = [x for x in s if x in choices]
            hit_words = set(hit_words)
            if len(hit_words) > 1:
                return True
            if choices[k] in string.ascii_uppercase:
                has_label = True
        return has_label

    assert input_file.endswith('.tsv')
    data = load(input_file)
    n_opt = 2
    for i, ch in enumerate(string.ascii_uppercase):
        if ch in data:
            n_opt = ord(ch) - ord('A') + 1
        else:
            for j in range(i + 1, 26):
                assert string.ascii_uppercase[j] not in data
    groups = defaultdict(list)
    for i in range(len(data)):
        item = data.iloc[i]
        this_n_opt = 0
        for j, ch in enumerate(string.ascii_uppercase[:n_opt]):
            if not pd.isna(item[ch]):
                this_n_opt = j + 1
            else:
                for k in range(j + 1, n_opt):
                    assert pd.isna(item[string.ascii_uppercase[k]]), (k, item)
        assert this_n_opt >= 2 or this_n_opt == 0
        flag = abnormal_entry(item)
        if flag or this_n_opt == 0:
            groups['abnormal'].append(item)
        elif len(item['answer']) > 1 or item['answer'] not in string.ascii_uppercase[:this_n_opt]:
            groups['abnormal'].append(item)
        else:
            groups[this_n_opt].append(item)
    for k in groups:
        groups[k] = pd.concat(groups[k], axis=1).T
        print(f'{k if k == "abnormal" else str(k) + "-choice"} records: {len(groups[k])}')

    data_all = []

    for k in groups:
        if k == 'abnormal':
            warnings.warn(
                f"{len(groups['abnormal'])} abnormal entries detected. The problems can be: "
                "1. Choice labels found in some choice contents; 2. No choices found for this question; "
                "3. The answer is not a valid choice. Will not apply circular to those samples."
            )
            abdata = groups['abnormal']
            abdata['g_index'] = abdata['index']
            data_all.append(abdata)
        else:
            cir_data = []
            assert isinstance(k, int) and k >= 2
            labels = string.ascii_uppercase[:k]
            rotates = [labels]
            dq = deque(labels)
            for i in range(k - 1):
                dq.rotate(1)
                rotates.append(list(dq))
            for i, rot in enumerate(rotates):
                if i == 0:
                    data = groups[k].copy()
                    data['g_index'] = data['index']
                    cir_data.append(data)
                else:
                    try:
                        data = groups[k].copy()
                        data['g_index'] = [x for x in data['index']]
                        data['index'] = [x + f'__group_{i}' for x in data['index']]
                        data['image'] = data['g_index']
                        c_map = {k: v for k, v in zip(rotates[0], rot)}
                        data['answer'] = [c_map[x] for x in data['answer']]
                        for s, t in c_map.items():
                            data[t] = groups[k][s]
                        cir_data.append(data)
                    except:
                        print(set(data['answer']))
                        raise NotImplementedError
            data_all.append(pd.concat(cir_data))
    data_all = pd.concat(data_all)

    tgt_file = input_file.replace('.tsv', '_circular.tsv')
    dump(data_all, tgt_file)
    print(f'Processed data are saved to {tgt_file}: {len(load(input_file))} raw records, {len(data_all)} circularized records.')  # noqa: E501
    assert osp.exists(tgt_file)
    print(f'The MD5 for the circularized data is {md5(tgt_file)}')
