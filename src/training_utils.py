#!/usr/bin/python
"""utility functions for training models"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def train_vae(vae, optimizer, dataloader):
    """training a vae on a single iteration of data"""
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
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


def train_classifier(classifier, optimizer, dataloader):
    """training a classifer on a single iteration of data"""
    classifier.train()
    criterion = nn.CrossEntropyLoss()
    train_loss = 0
    correct = 0 # for later accuracy calculation. Entirely for evaluation
    for data, target in dataloader: # classification problem
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


def generate_digits(vae, num_samples, vae_latent_dim):
    """generate new digits"""
    vae.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, vae_latent_dim)
        samples = vae.decoder(z)
    return samples


def classify_digits(classifier, digits):
    """classifying digits"""
    classifier.eval()
    with torch.no_grad():
        output = classifier(digits)
        labels = output.argmax(dim=1)
    return labels


def evaluate_classifier(classifier, dataloader):
    """evaluating the performance of a given classifier on a dataloader"""
    classifier.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0 
    correct = 0 
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            output = classifier(data.view(data.size(0), -1))
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

