#!/usr/bin/python3
"""Configuration management for MNIST model collapse experiments"""

import yaml
from pathlib import Path


def load_config(config_path):
    """load configuration from YAML file"""
    config_path = Path(config_path)

    if not config_path or not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def merge_config_with_cli(config, cli_args):
    """overwriting config values with those specified in CLI"""
    # Map CLI argument names to config paths
    cli_to_config_mapping = {
        'iterations': ('experiment', 'iterations'),
        'batch_size': ('experiment', 'batch_size'),
        'vae_lr': ('vae', 'learning_rate'),
        'vae_epochs': ('vae', 'epochs'),
        'classifier_lr': ('classifier', 'learning_rate'),
        'classifier_epochs': ('classifier', 'epochs'),
        'num_generated_samples': ('experiment', 'num_generated_samples'),
        'num_displayed_samples': ('experiment', 'num_displayed_samples'),
    }

    # Override config values with CLI arguments
    for cli_key, config_path in cli_to_config_mapping.items():
        if cli_key in cli_args and cli_args[cli_key] is not None:
            # Navigate to nested dict and set value
            current = config
            for key in config_path[:-1]:
                current = current[key]
            current[config_path[-1]] = cli_args[cli_key]

    return config


def get_config_value(config, *keys):
    """Safely get nested config value with a list of keys"""
    current = config
    for key in keys:
        current = current[key]
    return current
