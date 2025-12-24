#!/usr/bin/python
"""utility functions to visualise the outputs of runs"""

import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


def visualise_digits(digits, labels, iteration):
    """randomly sample from the input dataset and visualise the samples"""
    num_digits = len(digits)
    num_rows = math.ceil(num_digits / 10) # allows arbitrarily many images to be shown. Displays these images in rows of 10
    num_cols = min(num_digits, 10)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2*num_cols, 2*num_rows))
    fig.suptitle(f'Generated Digits - Iteration {iteration}')

    if num_rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_digits):
        row = i // 10
        col = i % 10
        ax = axes[row, col]
        ax.imshow(digits[i].view(28, 28).numpy(), cmap='gray') # recreates original MNIST images as faithfully as possible
        ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')

    # Remove any unused subplots
    for i in range(num_digits, num_rows * num_cols):
        row = i // 10
        col = i % 10
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    return fig


def colour_plot_matrix(matrix, title, cmap='viridis', num_ticks=11):
    """creates a colour plot of a matrix, not displaying explicit values"""
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Dataset Iteration')
    plt.ylabel('Classifier Iteration')
    
    # Create evenly spaced tick locations
    x_ticks = np.linspace(0, matrix.shape[1] - 1, num_ticks, dtype=int)
    y_ticks = np.linspace(0, matrix.shape[0] - 1, num_ticks, dtype=int)
    
    # Set tick locations and labels
    plt.xticks(x_ticks, x_ticks + 1)
    plt.yticks(y_ticks, y_ticks + 1)
    
    return fig  # Return the figure object instead of showing

def create_samples_gif(run_dir, filename='model_collapse.gif', duration=200, loop=True):
    """create a gif of sampled images"""
    samples_dir = run_dir / "samples" # hardcoded samples directory
    gif_filename = run_dir / filename
    image_files = [] # image file paths

    for filename in sorted(os.listdir(samples_dir)):
        image_files.append(os.path.join(samples_dir, filename))

    if not image_files:
        print("Warning: No image files found in {samples_dir}. No gif created")
        return None

    # load images
    images = [Image.open(img) for img in image_files]

    images[0].save(
        gif_filename,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=False
    )

