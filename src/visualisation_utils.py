#!/usr/bin/python
"""utility functions to visualise the outputs of runs"""

import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import re


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

def plot_training_curves(matrix, title, xlabel='Epoch', ylabel='Value', max_curves=10, cmap='viridis'):
    """Plot training curves showing progression over epochs for each iteration"""
    fig, ax = plt.subplots(figsize=(12, 6))

    num_iterations = matrix.shape[0]

    # If too many iterations, sample them evenly
    if num_iterations > max_curves:
        indices = np.linspace(0, num_iterations - 1, max_curves, dtype=int)
    else:
        indices = range(num_iterations)

    # Create colormap
    colormap = plt.get_cmap(cmap)
    colors = [colormap(i / (len(indices) - 1)) for i in range(len(indices))]

    # Plot each iteration's training curve with gradient colors
    for i, idx in enumerate(indices):
        ax.plot(matrix[idx, :], alpha=0.8, color=colors[i], label=f'Iteration {idx + 1}', linewidth=1.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def save_training_plots(run_dir, vae_loss_matrix, classifier_loss_matrix, classifier_accuracy_matrix):
    """Save all training curve plots to a subdirectory in run_dir"""
    # Create training_curves subdirectory
    plots_dir = run_dir / "plots" / "training_curves"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot and save VAE training loss
    fig = plot_training_curves(
        vae_loss_matrix,
        title='VAE Training Loss Over Epochs',
        xlabel='Epoch',
        ylabel='Loss'
    )
    fig.savefig(plots_dir / 'vae_training_loss.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Plot and save classifier training loss
    fig = plot_training_curves(
        classifier_loss_matrix,
        title='Classifier Training Loss Over Epochs',
        xlabel='Epoch',
        ylabel='Loss'
    )
    fig.savefig(plots_dir / 'classifier_training_loss.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Plot and save classifier training accuracy
    fig = plot_training_curves(
        classifier_accuracy_matrix,
        title='Classifier Training Accuracy Over Epochs',
        xlabel='Epoch',
        ylabel='Accuracy'
    )
    fig.savefig(plots_dir / 'classifier_training_accuracy.png', bbox_inches='tight', dpi=300)
    plt.close(fig)


def create_samples_gif(run_dir, filename='model_collapse.gif', duration=200, loop=True):
    """create a gif of sampled images"""
    samples_dir = run_dir / "samples" # hardcoded samples directory
    gif_filename = run_dir / filename
    image_files = [] # image file paths

    # Sort by numeric iteration number, not alphabetically
    def get_iteration_number(fname):
        """Extract iteration number from filename like 'iteration_{val}.png'"""
        match = re.search(r'iteration_(\d+)', fname)
        return int(match.group(1)) if match else 0

    for filename in sorted(os.listdir(samples_dir), key=get_iteration_number):
        image_files.append(os.path.join(samples_dir, filename))

    if not image_files:
        print("Warning: No image files found in {samples_dir}. No gif created")
        return None

    # load images
    images = [Image.open(img) for img in image_files]

    if loop:
        loop = 0 # for infinite loop
    else:
        loop = 1 # for no looping

    images[0].save(
        gif_filename,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=False
    )


def plot_final_values(matrix, title, xlabel='Iteration', ylabel='Value'):
    """plotting the final loss and accuracy from different runs"""
    fig, ax = plt.subplots(figsize=(12, 6))

    final_values = matrix[:, -1] # last column for a 2D array
    xvalues = np.arange(len(final_values))

    ax.plot(xvalues, final_values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig

def save_final_values(run_dir, vae_loss_matrix, classifier_loss_matrix, classifier_accuracy_matrix):
    """saving all the final values plots in run_dir"""
    # creating final_values directory
    final_values_dir = run_dir / "plots" / "final_values"
    final_values_dir.mkdir(parents=True, exist_ok=True)

    # plot and save the VAE final loss values
    fig = plot_final_values(
        vae_loss_matrix, title='VAE Final Training Loss Over Iterations',
        xlabel='Iteration',
        ylabel='Loss'
    )

    fig.savefig(final_values_dir / "vae_final_loss.png", bbox_inches='tight', dpi=300)
    plt.close(fig)

    # plot and save the clasifier final training loss
    fig = plot_final_values(classifier_loss_matrix, title='Classifier Final Training Loss Over Iterations', xlabel='Iteration', ylabel='Loss')

    fig.savefig(final_values_dir / "classifier_final_loss.png", bbox_inches='tight', dpi=300)
    plt.close(fig)

    # plot and save the classifier final accuracies
    fig = plot_final_values(classifier_accuracy_matrix, title='Classifier Final Accuracy Over Iterations', xlabel='Iteration', ylabel='Accuracy')

    fig.savefig(final_values_dir / "classifier_final_accuracy.png", bbox_inches='tight', dpi=300)
    plt.close(fig)


def save_label_frequency(run_dir, label_counts, title, filename, cmap='viridis'):
    """save the frequency of different labels in the generated datasets"""
    label_freq_dir = run_dir / "plots" / "label_frequencies"
    label_freq_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    # iterations, where the first value represents the initial dataset and is set to 0
    xvalues = np.arange(len(label_counts))

    # finding all unique indices in the list of dictionaries
    indices = list(set([y for sublist in [x.keys() for x in label_counts] for y in sublist]))

    colormap = plt.get_cmap(cmap)
    colors = [colormap(i / (len(indices) - 1)) for i in range(len(indices))]

    for value, idx in enumerate(indices):
        value_counts = []
        for freq_dict in label_counts:
            value_counts.append(freq_dict.get(value, 0))
        ax.plot(xvalues, value_counts, color=colors[idx], label=f'Counts of {value}', linewidth=1.5)
    
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Label Counts")
    ax.set_title("Distribution Of Label Counts Over Iterations")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig.savefig(label_freq_dir / filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

