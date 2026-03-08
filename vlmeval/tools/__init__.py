import click
from .eval import EVAL, PRINT_ACC, RUN
from .circular import CIRCULAR
from .localize import LOCALIZE
from .dataset_cmds import DLIST
from .check import CHECK
from .merge_pkl import MERGE_PKL
from .model_cmds import MLIST
from .missing import FIND_MISSING
from .fetch_res import FETCH

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass


cli.add_command(EVAL, name='eval')
cli.add_command(CIRCULAR, name='circular')
cli.add_command(LOCALIZE, name='localize')
cli.add_command(DLIST, name='dlist')
cli.add_command(CHECK, name='check')
cli.add_command(MERGE_PKL, name='merge_pkl')
cli.add_command(MLIST, name='mlist')
cli.add_command(FIND_MISSING, name='missing')
cli.add_command(PRINT_ACC, name='print_acc')
cli.add_command(FETCH, name='fetch')
cli.add_command(RUN, name='run')
