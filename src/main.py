#!/usr/bin/python3
"""experimenting with model collapse on MNIST"""

import torch
import torch.optim as optim
import numpy as np
import click
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import yaml

# local imports
from load_dataset import load_MNIST
from models import VAE, Classifier
from training_utils import train_vae, train_classifier, generate_digits, classify_digits, evaluate_classifier, label_stats_find
from visualisation_utils import visualise_digits, colour_plot_matrix, create_samples_gif, save_training_plots, save_final_values, save_label_frequency
from config import load_config, merge_config_with_cli
from utils import set_random_seed


def load_datasets():
    """loading the datasets"""
    # data directory
    data_dir = Path("../data/")

    # importing the datasets
    train_dataset, test_dataset = load_MNIST(download=True, data_dir=data_dir)

    return train_dataset, test_dataset


def initialise_vae(config):
    """initialising the vae"""
    vae_layer_sizes = config['vae']['layer_sizes']
    vae_latent_dim = config['vae']['latent_dim']
    vae = VAE(vae_layer_sizes, vae_latent_dim)
    return vae, vae_latent_dim


def initialise_classifier(config):
    """initialising a classifier"""
    classifier_layer_sizes = config['classifier']['layer_sizes']
    num_classes = config['classifier']['num_classes']
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


def detect_device():
    """Detect available hardware acceleration (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def set_num_workers(device):
    """Setting the number of workers to be used for data loading"""
    # Use multiprocessing only for CUDA
    # MPS and CPU have pickling issues with lambda transforms on macOS
    num_workers = 4 if device.type == 'cuda' else 0
    return num_workers


def setup_experiment(cfg, output_dir=None):
    """setup experiment directories and save configuration"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiments_dir = Path('../experiments/')

    if output_dir:
        run_dir = experiments_dir / output_dir
    else:
        run_dir = experiments_dir / f"run_{timestamp}"

    run_dir.mkdir(parents=True, exist_ok=True)

    # Save the config yaml file to the run_dir for reproducibility
    config_file_path = run_dir / "config.yaml"
    with open(config_file_path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, allow_unicode=True)

    return run_dir


def initialise_experiment_data(cfg, train_dataset):
    """initialise all data structures needed for the experiment"""
    iterations = cfg['experiment']['iterations']
    vae_epochs = cfg['vae']['epochs']
    classifier_epochs = cfg['classifier']['epochs']

    # Preallocate matrices
    loss_matrix = np.zeros((iterations, iterations))
    accuracy_matrix = np.zeros((iterations, iterations))
    vae_loss_matrix = np.zeros((iterations, vae_epochs))
    classifier_loss_matrix = np.zeros((iterations, classifier_epochs))
    classifier_accuracy_matrix = np.zeros((iterations, classifier_epochs))

    # Initialise label tracking
    label_counts = []
    labels_list = [y for (x, y) in train_dataset]
    initial_count_dict = label_stats_find(torch.Tensor(labels_list))
    label_counts.append(initial_count_dict)
    initial_label_counts = []

    return {
        'loss_matrix': loss_matrix,
        'accuracy_matrix': accuracy_matrix,
        'vae_loss_matrix': vae_loss_matrix,
        'classifier_loss_matrix': classifier_loss_matrix,
        'classifier_accuracy_matrix': classifier_accuracy_matrix,
        'label_counts': label_counts,
        'initial_label_counts': initial_label_counts,
        'classifiers': []
    }


def run_training_iteration(iteration, vae, cfg, current_dataset, device, data_structs):
    """run a single iteration of the training loop"""
    # extracting parameters
    batch_size = cfg['experiment']['batch_size']
    vae_lr = cfg['vae']['learning_rate']
    vae_epochs = cfg['vae']['epochs']
    classifier_lr = cfg['classifier']['learning_rate']
    classifier_epochs = cfg['classifier']['epochs']
    num_generated_samples = cfg['experiment']['num_generated_samples']
    vae_latent_dim = cfg['vae']['latent_dim']

    # create DataLoader
    num_workers = set_num_workers(device)
    dataloader = DataLoader(current_dataset, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)

    # train VAE and storing training loss
    vae_optimizer = optim.Adam(vae.parameters(), lr=vae_lr)
    vae_pbar = tqdm(range(vae_epochs), desc='  Training VAE', leave=False, dynamic_ncols=True)
    for epoch in vae_pbar:
        loss = train_vae(vae, vae_optimizer, dataloader, device)
        data_structs['vae_loss_matrix'][iteration - 1, epoch] = loss
        vae_pbar.set_postfix({'loss': f'{loss:.4f}'})

    # initialise and train new classifier
    classifier = initialise_classifier(cfg)
    classifier = classifier.to(device)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=classifier_lr)
    classifier_pbar = tqdm(range(classifier_epochs), desc='  Training Classifier', leave=False, dynamic_ncols=True)
    for epoch in classifier_pbar:
        loss, accuracy = train_classifier(classifier, classifier_optimizer, dataloader, device)
        data_structs['classifier_loss_matrix'][iteration - 1, epoch] = loss
        data_structs['classifier_accuracy_matrix'][iteration - 1, epoch] = accuracy
        classifier_pbar.set_postfix({'loss': f'{loss:.4f}', 'accuracy': f'{accuracy:.4f}'})

    # store the trained classifier for later evaluation
    data_structs['classifiers'].append(classifier)

    # generate new dataset
    generated_digits = generate_digits(vae, num_generated_samples, vae_latent_dim, device)
    generated_labels = classify_digits(classifier, generated_digits, device)

    # track label counts
    label_counts_dict = label_stats_find(generated_labels)
    data_structs['label_counts'].append(label_counts_dict)

    new_dataset = TensorDataset(generated_digits, generated_labels)

    return new_dataset, classifier, generated_digits


def evaluate_all_classifiers(data_structs, current_dataset, iteration, batch_size, device):
    """evaluate all previous classifiers on the current dataset"""
    # create dataloader from current dataset
    num_workers = set_num_workers(device)
    dataloader = DataLoader(current_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)

    # evlauate all previously trained classifiers on this data
    for prev_iteration, prev_classifier in enumerate(data_structs['classifiers']):
        loss, accuracy = evaluate_classifier(prev_classifier, dataloader, device)
        data_structs['loss_matrix'][prev_iteration, iteration - 1] = loss
        data_structs['accuracy_matrix'][prev_iteration, iteration - 1] = accuracy


def save_experiment_results(run_dir, data_structs):
    """save all experiment results and visualizations"""
    # save analysis matrices
    save_analysis_matrices(run_dir, data_structs['loss_matrix'], 'Loss Matrix', 'loss_matrix.png')
    save_analysis_matrices(run_dir, data_structs['accuracy_matrix'], 'Accuracy Matrix', 'accuracy_matrix.png')

    # save training curves
    save_training_plots(run_dir, data_structs['vae_loss_matrix'],
                       data_structs['classifier_loss_matrix'],
                       data_structs['classifier_accuracy_matrix'])

    # save final loss and accuracy values
    save_final_values(run_dir, data_structs['vae_loss_matrix'],
                     data_structs['classifier_loss_matrix'],
                     data_structs['classifier_accuracy_matrix'])

    # save label frequency distributions
    save_label_frequency(run_dir, data_structs['label_counts'],
                        filename="label_counts.png",
                        title="Distribution Of Label Counts Over Iterations")

    save_label_frequency(run_dir, data_structs['initial_label_counts'],
                        filename="initial_label_counts.png",
                        title="Distribution Of Label Counts Over Iterations (using the initial classifier)")

    # create gif from saved sample images, and save
    create_samples_gif(run_dir, filename='model_collapse.gif')


@click.command(context_settings=dict(help_option_names=['-h', '--help'], show_default=True))
@click.option('--config', '-c', help='Path to config file (YAML). CLI args override config values.', default="../config.yaml", type=click.Path(exists=True))
@click.option('-i', '--iterations', help='number of iterations of VAE training and generation', default=None, type=int)
@click.option('-b', '--batch-size', help='batch size for VAE training', default=None, type=int)
@click.option('--vae-lr', help='VAE learning rate', default=None, type=float)
@click.option('--vae-epochs', help='VAE training epochs', default=None, type=int)
@click.option('--classifier-lr', help='Classifier learning rate', default=None, type=float)
@click.option('--classifier-epochs', help='Classifer training epochs', default=None, type=int)
@click.option('--num-generated-samples', help='Number of samples to be generated per iteration', default=None, type=int)
@click.option('--num-displayed-samples', help='Number of samples from an iteration to be displayed', default=None, type=int)
@click.option('--random-seed', help='Random seed for reproducibility (use -1 to disable)', default=None, type=int)
@click.option('--output-dir', help='directory to save outputs in (inside analysis/)', default=None, type=str)
def main(config, iterations, batch_size, vae_lr, vae_epochs, classifier_lr, classifier_epochs, num_generated_samples, num_displayed_samples, random_seed, output_dir):
    """Main script for MNIST model collapse experiments"""

    # Load and merge configuration
    cfg = load_config(config)
    cli_args = {
        'iterations': iterations,
        'batch_size': batch_size,
        'vae_lr': vae_lr,
        'vae_epochs': vae_epochs,
        'classifier_lr': classifier_lr,
        'classifier_epochs': classifier_epochs,
        'num_generated_samples': num_generated_samples,
        'num_displayed_samples': num_displayed_samples,
        'random_seed': random_seed,
    }
    cfg = merge_config_with_cli(cfg, cli_args)

    # Extract frequently-used values from config
    iterations = cfg['experiment']['iterations']
    batch_size = cfg['experiment']['batch_size']
    num_displayed_samples = cfg['experiment']['num_displayed_samples']
    random_seed = cfg['experiment']['random_seed']

    # Set random seed for reproducibility
    if random_seed is not None and random_seed >= 0:
        print(f"Setting random seed to {random_seed} for reproducibility. Note that this may degrade performance")
        set_random_seed(random_seed)
    else:
        print("Running without fixed random seed (results will not be reproducible)")

    # Setup device
    device = detect_device()
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif device.type == 'mps':
        print("Using Apple Silicon GPU (MPS)")
        print("PyTorch MPS backend enabled")

    # Setup experiment directory and save config
    run_dir = setup_experiment(cfg, output_dir)
    print(f"Experiment outputs will be saved to: {run_dir}")

    # Load datasets and initialise models
    train_dataset, test_dataset = load_datasets()
    vae, vae_latent_dim = initialise_vae(cfg)
    vae = vae.to(device)

    # Initialise experiment data structures
    data_structs = initialise_experiment_data(cfg, train_dataset)
    current_dataset = train_dataset

    # Save initial samples
    num_workers = set_num_workers(device)
    initial_dataloader = DataLoader(current_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
    save_data_samples(run_dir, 0, initial_dataloader, num_displayed_samples)

    # Main training loop
    initial_classifier = None
    pbar = tqdm(range(1, iterations + 1), desc='Dataset Iterations', leave=True, dynamic_ncols=True)
    for iteration in pbar:
        # Run training iteration
        current_dataset, classifier, generated_digits = run_training_iteration(
            iteration, vae, cfg, current_dataset, device, data_structs
        )

        # Save the first classifier for later analysis
        if iteration == 1:
            initial_classifier = classifier
            digits_list = [x for (x, _) in train_dataset]
            digits_tensor = torch.stack(digits_list, dim=0)
            initially_classified_labels = classify_digits(initial_classifier, digits_tensor, device)
            label_counts_dict = label_stats_find(initially_classified_labels)
            data_structs['initial_label_counts'].append(label_counts_dict)

        # Evaluate all previous classifiers on the new dataset
        evaluate_all_classifiers(data_structs, current_dataset, iteration, batch_size, device)

        # Save image samples
        num_workers = set_num_workers(device)
        new_dataloader = DataLoader(current_dataset, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
        save_data_samples(run_dir, iteration, new_dataloader, num_displayed_samples)

        # Track labels from initial classifier's perspective
        initially_classified_labels = classify_digits(initial_classifier, generated_digits, device)
        label_counts_dict = label_stats_find(initially_classified_labels)
        data_structs['initial_label_counts'].append(label_counts_dict)

    # Save all experiment results
    save_experiment_results(run_dir, data_structs)
    print(f"Experiment complete! Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
