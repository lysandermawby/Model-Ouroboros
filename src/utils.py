#!/usr/bin/python3
"""Utility functions for experiment setup and config validation"""

import torch
import numpy as np
import random


def set_random_seed(seed = 0):
    """set random seeds for reproducibility across PyTorch, NumPy, and Python random"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    np.random.seed(seed)
    random.seed(seed) # used by some functions

    # For deterministic behavior in PyTorch (may impact performance)
    # only utilise if user wants to set a deterministic seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_config(config):
    """validate configuration values to ensure they are sensible"""
    # Validate experiment settings
    _validate_experiment_config(config.get('experiment', {}))

    # Validate VAE settings
    _validate_vae_config(config.get('vae', {}))

    # Validate classifier settings
    _validate_classifier_config(config.get('classifier', {}))

    # Validate visualization settings
    _validate_visualization_config(config.get('visualization', {}))


def _validate_experiment_config(exp_config):
    """Validate experiment configuration section"""
    if 'iterations' in exp_config:
        iterations = exp_config['iterations']
        if not isinstance(iterations, int) or iterations < 1:
            raise ValueError(f"iterations must be a positive integer, got: {iterations}")

    if 'batch_size' in exp_config:
        batch_size = exp_config['batch_size']
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got: {batch_size}")
        if batch_size > 10000:
            print(f"Batch size {batch_size} seems very high. This may cause memory overflows. Are you sure that you want to proceed with such a high value?")
            print("To change the batch size, either edit the experiments/batch_size field in the config.yaml file or run main.py with --batch-size")

    if 'num_generated_samples' in exp_config:
        num_samples = exp_config['num_generated_samples']
        if not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError(f"num_generated_samples must be a positive integer, got: {num_samples}")

    if 'num_displayed_samples' in exp_config:
        num_display = exp_config['num_displayed_samples']
        if not isinstance(num_display, int) or num_display < 1:
            raise ValueError(f"num_displayed_samples must be a positive integer, got: {num_display}")
        if num_display % 10 != 0:
            print(f"num_displayed_samples will be rounded down to a multiple of 10. Got: {num_display}")
            print(f"The display function will remove images, only displaying {num_display - (num_display % 10)}")


def _validate_vae_config(vae_config) -> None:
    """Validate VAE configuration section"""
    if 'layer_sizes' in vae_config:
        layer_sizes = vae_config['layer_sizes']
        if not isinstance(layer_sizes, list) or len(layer_sizes) < 1:
            raise ValueError(f"vae.layer_sizes must be a list with at least 1 element, got: {layer_sizes}")
        if layer_sizes[0] != 784:
            raise ValueError(f"First layer size must be 784 for MNIST (28x28 flattened), got: {layer_sizes[0]}")
        if not all(isinstance(size, int) and size > 0 for size in layer_sizes):
            raise ValueError(f"All layer sizes must be positive integers, got: {layer_sizes}")

    if 'latent_dim' in vae_config:
        latent_dim = vae_config['latent_dim']
        if not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError(f"vae.latent_dim must be a positive integer, got: {latent_dim}")

    if 'learning_rate' in vae_config:
        lr = vae_config['learning_rate']
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError(f"vae.learning_rate must be a positive number, got: {lr}")
        if lr > 0.1:
            print(f"vae.learning_rate seems very high ({lr}). Typical values are < 0.1")

    if 'epochs' in vae_config:
        epochs = vae_config['epochs']
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError(f"vae.epochs must be a positive integer, got: {epochs}")


def _validate_classifier_config(clf_config):
    """Validate classifier configuration section"""
    if 'layer_sizes' in clf_config:
        layer_sizes = clf_config['layer_sizes']
        if not isinstance(layer_sizes, list) or len(layer_sizes) < 1:
            raise ValueError(f"classifier.layer_sizes must be a list with at least 1 element, got: {layer_sizes}")
        if layer_sizes[0] != 784:
            raise ValueError(f"First layer size must be 784 for MNIST (28x28 flattened), got: {layer_sizes[0]}")
        if not all(isinstance(size, int) and size > 0 for size in layer_sizes):
            raise ValueError(f"All layer sizes must be positive integers, got: {layer_sizes}")

    if 'num_classes' in clf_config:
        num_classes = clf_config['num_classes']
        if not isinstance(num_classes, int):
            raise ValueError(f"classifier.num_classes must be an integer, got: {num_classes}")
        if num_classes != 10:
            raise ValueError(f"For MNIST, num_classes should be 10, got: {num_classes}")

    if 'learning_rate' in clf_config:
        lr = clf_config['learning_rate']
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError(f"classifier.learning_rate must be a positive number, got: {lr}")
        if lr > 0.1:
            print(f"classifier.learning_rate seems very high ({lr}). Typical values are < 0.1")

    if 'epochs' in clf_config:
        epochs = clf_config['epochs']
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError(f"classifier.epochs must be a positive integer, got: {epochs}")


def _validate_visualization_config(viz_config):
    """Validate visualization configuration section"""
    if 'sample_gif_duration' in viz_config:
        duration = viz_config['sample_gif_duration']
        if not isinstance(duration, int) or duration < 1:
            raise ValueError(f"visualization.sample_gif_duration must be a positive integer, got: {duration}")

    if 'training_curves_max_lines' in viz_config:
        max_lines = viz_config['training_curves_max_lines']
        if not isinstance(max_lines, int) or max_lines < 1:
            raise ValueError(f"visualization.training_curves_max_lines must be a positive integer, got: {max_lines}")

    if 'colormap' in viz_config:
        colormap = viz_config['colormap']
        if not isinstance(colormap, str):
            raise ValueError(f"visualization.colormap must be a string, got: {colormap}")
