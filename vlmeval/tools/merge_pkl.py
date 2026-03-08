import os
from vlmeval.smp import osp, load, dump, defaultdict
import click


@click.command()
@click.argument('pkl_dir', type=click.Path(exists=True))
@click.option('--world-size', type=int, default=1)
def MERGE_PKL(pkl_dir, world_size=1):
    prefs = []
    for ws in list(range(1, 9)):
        prefs.extend([f'{i}{ws}_' for i in range(ws)])
    prefs = set(prefs)
    files = os.listdir(pkl_dir)
    files = [x for x in files if x[:3] in prefs]
    # Merge the files
    res_all = defaultdict(dict)
    for f in files:
        full_path = osp.join(pkl_dir, f)
        key = f[3:]
        res_all[key].update(load(full_path))
        os.remove(full_path)

    dump_prefs = [f'{i}{world_size}_' for i in range(world_size)]
    for k in res_all:
        for pf in dump_prefs:
            dump(res_all[k], f'{pkl_dir}/{pf}{k}')
        print(f'Merged {len(res_all[k])} records into {pkl_dir}/{dump_prefs[0]}{k}')
