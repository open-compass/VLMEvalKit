import click
import os.path as osp
from vlmeval.smp import load, dump, localize_tsv


@click.command()
@click.argument('fname', type=str)
def LOCALIZE(fname):
    new_fname = fname.replace('.tsv', '_local.tsv')
    localize_tsv(fname, new_fname)
    return new_fname
