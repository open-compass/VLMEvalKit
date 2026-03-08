import click


@click.command()
@click.argument('category', type=click.Choice([
    'VLM', 'ULM', 'API']))
def MLIST(category):
    from vlmeval.config import supported_VLM, supported_ULM, supported_APIs
    if category == 'VLM':
        ret = [x for x in supported_VLM]
    elif category == 'ULM':
        ret = [x for x in supported_ULM]
    elif category == 'API':
        ret = [x for x in supported_APIs]
    print(' '.join(ret))
    return ret
