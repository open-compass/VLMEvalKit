import click
import os
import os.path as osp


@click.command()
@click.argument('name', type=str)
@click.option('--source', default=None, type=str)
@click.option('--target', default=None, type=str)
def FETCH(name, source=None, target=None):
    import vlmeval
    root = vlmeval.__path__[0] + '/../'
    if source is None:
        source = osp.join(root, 'outputs/')
    if target is None:
        target = osp.join(root, 'debug', name)
    os.makedirs(target, exist_ok=True)
    fetch_dataset(name, source, target)


def fetch_dataset(name, source, target):
    from vlmeval.smp import ls
    from vlmeval.dataset import SUPPORTED_DATASETS
    prefix_datasets = [
        x for x in SUPPORTED_DATASETS if x.startswith(name) and x != name
    ]
    files = ls(source, level=2, match=[name] + ['!' + x for x in prefix_datasets])
    for f in files:
        os.system(f'cp -Lr {f} {target}/')
