#!/usr/bin/python
"""creating a gif from a previously generated set of images"""

import click
from visualisation_utils import create_samples_gif
import os
from pathlib import Path


@click.command(context_settings=dict(help_option_names=['-h', '--help'], show_default=True))
@click.option('-r', '--run-dir', help='directory to create gif for', type=str)
@click.option('-f', '--filename', help='filename to save to (by default overwrites previous entry)', default='model_collapse.gif', type=str)
@click.option('-d', '--duration', help='Time spent on each frame (ms)', default=200, type=int)
@click.option('-l', '--loop', help='Create a gif that automatically loops', default=True, is_flag=True)
@click.option('--list', help='List all available run directories (prevents gif creation)', default=False, is_flag=True)
def main(run_dir, filename, duration, loop, list):
    """creating a gif from a previously generated set of images"""
    experiments_dir = Path("../experiments")
    if list:
        run_dirs = os.listdir(experiments_dir)
        run_dirs = [x for x in run_dirs if os.path.isdir(experiments_dir / x)]
        run_dirs.sort()
        print("All available run directories:")
        print('\n'.join(run_dirs))
    elif not list and not run_dir:
        print("Warning: One of run_dir and list must be provided as command line arguments")
        print("Run this script with -h or --help for more information")
    else:
        run_dir = experiments_dir / run_dir
        create_samples_gif(run_dir, filename=filename, duration=duration, loop=loop)
        print(f"Created samples gif, saved to {run_dir}/{filename}")


if __name__ == '__main__':
    main()
