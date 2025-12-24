#!/usr/bin/python
"""utility functions for training models"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def train_vae(vae, optimizer, dataloader, device):
    """training a vae on a single iteration of data"""
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)  # Move data to GPU
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data) # reconstructed batch straight from vae.forward
        loss = vae_loss_function(recon_batch, data, mu, logvar)
        loss.backward() # fails when reparameterisation isn't implemented appropriately
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(dataloader.dataset)


def vae_loss_function(recon_x, x, mu, logvar):
    """custom loss function for the VAE"""
    x_flat = x.view(x.size(0), -1)
    MSE = F.mse_loss(recon_x, x_flat, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # closed form for KLD between gaussians
    return MSE + KLD # can change weighting for more control. This seems to work fine for now


def train_classifier(classifier, optimizer, dataloader, device):
    """training a classifer on a single iteration of data"""
    classifier.train()
    criterion = nn.CrossEntropyLoss()
    train_loss = 0
    correct = 0 # for later accuracy calculation. Entirely for evaluation
    for data, target in dataloader: # classification problem
        data, target = data.to(device), target.to(device)  # Move data to GPU
        optimizer.zero_grad()
        output = classifier(data.view(data.size(0), -1))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # for accuracy evaluation only, not training
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    # returns: avg loss, avg accuracy
    return train_loss / len(dataloader.dataset), correct / len(dataloader.dataset)


def generate_digits(vae, num_samples, vae_latent_dim, device):
    """generate new digits"""
    vae.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, vae_latent_dim).to(device)  # Generate on GPU
        samples = vae.decoder(z)
    return samples.cpu()  # Move back to CPU for dataset creation


def classify_digits(classifier, digits, device):
    """classifying digits"""
    classifier.eval()
    digits = digits.to(device)

    with torch.no_grad():
        output = classifier(digits)
        labels = output.argmax(dim=1)
    return labels.cpu()  # Return labels on CPU for dataset creation


def evaluate_classifier(classifier, dataloader, device):
    """evaluating the performance of a given classifier on a dataloader"""
    classifier.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)  # Move data to GPU
            output = classifier(data.view(data.size(0), -1))
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

