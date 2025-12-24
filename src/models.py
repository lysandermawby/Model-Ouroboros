#!/usr/bin/python3
"""classes describing the models"""

import torch
import torch.nn as nn

"""variational autoencoder used to generate new samples"""
class VAE(nn.Module):
    def __init__(self, layer_sizes, latent_dim):
        # assumes that encoder and decoder are symmetric
        # latent-dim is the size of the central layer
        super(VAE, self).__init__()
        
        # Build encoder
        encoder_layers = []
        for i in range(len(layer_sizes) - 1):
            encoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            encoder_layers.append(nn.ReLU()) 
        
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space layers (mean and log variance)
        self.fc_mu = nn.Linear(layer_sizes[-1], latent_dim) # mean in latent dimension
        self.fc_logvar = nn.Linear(layer_sizes[-1], latent_dim) # log-variance

        # building decoder
        decoder_layer_sizes = [latent_dim] + layer_sizes[::-1]
        decoder_layers = []

        for i in range(len(decoder_layer_sizes) - 1):
            decoder_layers.append(nn.Linear(decoder_layer_sizes[i], decoder_layer_sizes[i+1]))
            
            if i < len(decoder_layer_sizes) - 2:  # ReLU for all but last layer
                decoder_layers.append(nn.ReLU())
            else:  # Sigmoid for output layer
                decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)

    def reparameterise(self, mu, logvar):
        """reparameterising the gaussian distribution"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """forward method of model"""
        # encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # reparameterise
        z = self.reparameterise(mu, logvar)
        
        # Decode
        reconstruction = self.decoder(z)
        
        return reconstruction, mu, logvar


"""basic linear classifier for MNIST digits"""
class Classifier(nn.Module):
    def __init__(self, layer_sizes, num_classes):
        # note that the layer_sizes must not include the final layer, which maps to the number of output classes
        # basic linear classifier using ReLU
        super(Classifier, self).__init__()

        model_layers = []
        for i in range(len(layer_sizes) - 1):
            model_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            model_layers.append(nn.ReLU())

        # Add final output layer
        model_layers.append(nn.Linear(layer_sizes[-1], num_classes))

        self.model = nn.Sequential(*model_layers)
    
    def forward(self, x):
        # extract logits
        return self.model(x)