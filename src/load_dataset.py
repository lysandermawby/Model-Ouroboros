#!/usr/bin/python
"""downloading the MNIST dataset to the data directory"""

from torchvision import datasets, transforms

def load_MNIST(download=True, data_dir='../data/'):
    """
    load the MNIST datasets

    can be run with the download option set to False to ensure only local downloads
    """

    # move pixel values to between 0-1, and flatten vector
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    # Download and load training data
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform
    )

    # Download and load test data
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transform
    )

    return train_dataset, test_dataset

# allowing this script to be run directly
if __name__ == '__main__':
    load_MNIST(download=True, data_dir='./data')
