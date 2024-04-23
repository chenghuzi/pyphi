from pathlib import Path

import click
from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def load_ctx_from_config(ctx, config):
    if Path(config).is_file():
        with open(config, 'r') as f:
            config = load(f.read(), Loader=Loader)
        ctx.default_map = config


@click.group()
@click.option('--config', default='config.yml', type=click.Path())
@click.pass_context
def cli(ctx, config):
    load_ctx_from_config(ctx, config)
