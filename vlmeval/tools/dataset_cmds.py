import click


@click.command()
@click.argument('name', type=str)
def DLIST(name):
    from vlmeval.config import DATASET_GROUPS
    if name in DATASET_GROUPS.keys():
        ret = DATASET_GROUPS[name]
    else:
        from vlmeval.dataset import SUPPORTED_DATASETS
        ret = SUPPORTED_DATASETS
    print(' '.join(ret))
    return ret
