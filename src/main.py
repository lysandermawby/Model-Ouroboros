#!/usr/bin/python3
"""experimenting with model collapse on MNIST"""

# import torch
# import torch.nn as nn
import torch.optim as optim
import numpy as np
import click
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# local imports
from load_dataset import load_MNIST
from models import VAE, Classifier
from training_utils import train_vae, train_classifier, generate_digits, classify_digits, evaluate_classifier
from visualisation_utils import visualise_digits, colour_plot_matrix, create_samples_gif


def load_datasets():
    """loading the datasets"""
    # data directory
    data_dir = Path("../data/")

    # importing the datasets
    train_loader, test_loader = load_MNIST(download=True, data_dir=data_dir)

    return train_loader, test_loader


def initialise_models():
    """initialising the VAE and classification models"""
    vae, vae_latent_dim = initialise_vae()
    classifier = initialise_classifier()

    return vae, classifier

def initialise_vae():
    """initialising the vae"""
    vae_layer_sizes = [784, 128] # note that the first layer must be 784, as this is the MNIST input dimension
    vae_latent_dim = 20
    vae = VAE(vae_layer_sizes, vae_latent_dim)
    return vae, vae_latent_dim


def initialise_classifier():
    """initialising a classifier"""
    classifier_layer_sizes = [784, 256]
    num_classes = 10 # set for MNIST
    classifier = Classifier(classifier_layer_sizes, num_classes)

    return classifier


def save_data_samples(run_dir, iteration, dataloader, num_to_save=10):
    """saving image samples"""
    samples_dir = run_dir / "samples/"
    samples_dir.mkdir(parents=True, exist_ok=True)

    images, labels = next(iter(dataloader))

    # Take first 10
    digits = list(images[:num_to_save])
    labels = labels[:num_to_save]

    fig = visualise_digits(digits, labels, iteration)
    
    figure_path = samples_dir / f"iteration_{iteration}.png"

    # Save to a custom path
    fig.savefig(figure_path, bbox_inches='tight', dpi=300)

    # free memory
    plt.close(fig)


def save_analysis_matrices(run_dir, matrix, title, file_name, cmap='viridis', num_ticks=11):
    """save matrices describing the loss and accuracy of previous classifiers on later datasets"""
    plots_dir = run_dir / "plots/"
    plots_dir.mkdir(parents=True, exist_ok=True)
    file_path = plots_dir / file_name

    # generating and saving the figure
    fig = colour_plot_matrix(matrix, title, cmap='viridis', num_ticks=11)
    fig.savefig(file_path, bbox_inches='tight', dpi=300)

    # free memory
    plt.close(fig)

    
@click.command(context_settings=dict(help_option_names=['-h', '--help'], show_default=True))
@click.option('-i', '--iterations', help='number of iterations of VAE training and generation', default=100, type=int)
@click.option('-b', '--batch-size', help='batch size for VAE training', default=128, type=int)
@click.option('--vae-lr', help='VAE learning rate', default=1e-3, type=float)
@click.option('--vae-epochs', help='VAE training epochs', default=200, type=int)
@click.option('--classifier-lr', help='Classifier learning rate', default=1e-3, type=float)
@click.option('--classifier-epochs', help='Classifer training epochs', default=200, type=int)
@click.option('--num-generated_samples', help='Number of samples to be generated per iteration', default=60000, type=int)
@click.option('--num-displayed-samples', help='Number of samples from an iteration to be displayed', default=10, type=int)
@click.option('--output-dir', help='directory to save outputs in (inside analysis/)', default='run_\{count\}/', type=str)
def main(iterations, batch_size, vae_lr, vae_epochs, classifier_lr, classifier_epochs, num_generated_samples, num_displayed_samples, output_dir):
    """main script logic"""

    # directory to save script outputs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiments_dir = Path('../experiments/')
    run_dir = experiments_dir / f"run_{timestamp}"

    # loading MNIST datasets
    train_dataset, test_dataset = load_datasets()

    # initialising the models
    vae, vae_latent_dim = initialise_vae()

    current_dataset = train_dataset # only using the training dataset
    classifiers = [] # classifiers are saved to record accuracy over time
    # preallocating the loss and accuracy matrices
    loss_matrix = np.zeros((iterations, iterations))
    accuracy_matrix = np.zeros((iterations, iterations))

    # store some initial samples
    initial_dataloader = DataLoader(current_dataset, batch_size=batch_size, shuffle=True)
    save_data_samples(run_dir, 0, initial_dataloader, num_displayed_samples)

    # main training loop
    pbar = tqdm(range(1, iterations + 1), desc='Dataset Iterations', leave=True, dynamic_ncols=True)
    for iteration in pbar:
        # Create DataLoader
        dataloader = DataLoader(current_dataset, batch_size=batch_size, shuffle=True)

        vae_optimizer = optim.Adam(vae.parameters(), lr=vae_lr)
        vae_pbar = tqdm(range(vae_epochs), desc='  Training VAE', leave=False, dynamic_ncols=True)
        for epoch in vae_pbar:
            loss = train_vae(vae, vae_optimizer, dataloader)
            vae_pbar.set_postfix({'loss': f'{loss:.4f}'})
            
        # Initialize and train classifier. Note that the classifier is reinitialised every training loop
        classifier = initialise_classifier()
        classifier_optimizer = optim.Adam(classifier.parameters(), lr=classifier_lr)
        classifier_pbar = tqdm(range(classifier_epochs), desc='  Training Classifier', leave=False, dynamic_ncols=True)
        for epoch in classifier_pbar:
            loss, accuracy = train_classifier(classifier, classifier_optimizer, dataloader)
            classifier_pbar.set_postfix({'loss': f'{loss:.4f}', 'accuracy': f'{accuracy:.4f}'})

        # Store the trained classifier in classifiers list. Used later for overall evaluation
        classifiers.append(classifier)

        # Generate new dataset
        generated_digits = generate_digits(vae, num_generated_samples, vae_latent_dim)
        generated_labels = classify_digits(classifier, generated_digits) # from prev classifier notably

        current_dataset = TensorDataset(generated_digits, generated_labels) # next iteration dataset

        # Evaluate all previous classifiers on the new dataset, and writing to the loss and accuracy arrays
        new_dataloader = DataLoader(current_dataset, batch_size=batch_size, shuffle=False)
        for prev_iteration, prev_classifier in enumerate(classifiers):
            loss, accuracy = evaluate_classifier(prev_classifier, new_dataloader)
            loss_matrix[prev_iteration, iteration - 1] = loss
            accuracy_matrix[prev_iteration, iteration - 1] = accuracy

        # save image samples
        save_data_samples(run_dir, iteration, new_dataloader, num_displayed_samples)

        # This line can be used to save the current dataset for later analysis
        #torch.save(current_dataset, f'generated_dataset_iteration_{iteration + 1}.pt')

    # generating visualisations of previous losses and accuracy
    save_analysis_matrices(run_dir, loss_matrix, 'Loss Matrix', 'loss_matrix.png')
    save_analysis_matrices(run_dir, accuracy_matrix, 'Accuracy Matrix', 'accuracy_matrix.png')

    # make a gif from the saved sample images
    create_samples_gif(run_dir, filename='model_collapse.gif')


if __name__ == "__main__":
    main()
