# Model Ouroboros

Demonstrating model collapse on the MNIST dataset.

Model Collapse refers to the degradation in model quality when trained on its own outputs. Given the importance of LLMs in frontier AI development and the now ubiquitous presence of LLM outputs in internet text archives, this phenomenon could slow or otherwise severely disrupt the creation of new foundation models over the next couple of years

## Quick Start

Download this repository using `git clone`.

```bash
git clone https://github.com/lysandermawby/Model-Ouroboros.git
```

To install relevant dependencies, run the `setup.sh` script.
Note that this repository assumes that you have [uv package management](https://docs.astral.sh/uv/getting-started/installation/) available. Despite this, a `requirements.txt` file is made available for users of older package management systems.

```bash
# if you have uv installed
chmod +x setup.sh
./setup.sh
# if you want to use pip instead
pip install -r requirements.txt
```

The core functionality, including downloading data, training and sampling from models, and visualising data can all be accessed through the `src/main.py` script.

```bash
cd src
uv run python main.py
```

## Training Process

A simple Variational Autoencoder (VAE) is trained on the MNIST dataset. A simple classifier, defined as a linear network, is also trained on the dataset.

The VAE then has 60,000 samples drawn (the same number as are in the training dataset of MNIST) to form an entirely new dataset. This new dataset is used as further training data for the VAE, and an entirely new classifier is trained on this new dataset.

The same VAE is continually trained on its own outputs, and sampled from each iteration. As can be clearly seen, the model not only gradually ceases to produce images which look like original MNIST samples, but the outputs look more and more similar over time. Eventually, the outputs almost always start to look like blurred 8s, and stop evolving.

Model collapse can be seen by looking at the samples degrade and converge over time. It can be somewhat quantified by taking the classifiers trained on each dataset iteration, and evaluating them on one anothers datasets. The later classifiers show no ability to classify the real MNIST dataset while the earlier classifiers retain some ability, implying that the dataset simplifies rather than entirely losing any resemblance to the original data.
