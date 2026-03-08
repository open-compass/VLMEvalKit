import click
import os
import os.path as osp


@click.command()
@click.argument('dname', type=str)
@click.option('-g', '--group', type=str)
def FIND_MISSING(dname, group):
    if dname[-1] in ['/', '\\']:
        dname = dname[:-1]
    model_name = osp.basename(dname)
    from vlmeval.config import DATASET_GROUPS
    assert group in DATASET_GROUPS, f'group {group} not found in DATASET_GROUPS'
    dataset_names = DATASET_GROUPS[group]
    bad_datasets = []
    for data in dataset_names:
        if not osp.exists(f'{dname}/{model_name}_{data}.tsv'):
            bad_datasets.append(data)
    if len(bad_datasets):
        print(' '.join(bad_datasets))
    else:
        print(f'All datasets in group {group} are completed.')
