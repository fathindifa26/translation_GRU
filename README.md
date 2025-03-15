# Seq2Seq Translation: Javanese to Indonesian

## Overview

This project implements a Sequence-to-Sequence (Seq2Seq) neural machine translation (NMT) model to translate text from Javanese (Jawa) to Indonesian. Built using PyTorch, the model utilizes an encoder-decoder architecture with Long Short-Term Memory (LSTM) or Gated Recurrent Unit (GRU) networks, commonly used for sequence-to-sequence tasks in natural language processing (NLP). The project includes scripts for data preprocessing, model training, inference, and evaluation, designed with modularity and extensibility in mind.

The dataset consists of 800 training pairs and 100 validation pairs of Javanese-Indonesian sentences. The model is evaluated using loss and perplexity (PPL) metrics during training, with inference capabilities to translate new Javanese sentences into Indonesian.

## Features
- **Seq2Seq Architecture**: Implements a custom encoder-decoder model with LSTM or GRU units, configurable for hidden dimensions and layers.
- **Training Pipeline**: Includes training, validation, and early stopping based on validation loss.
- **Inference**: Supports translation of new Javanese sentences into Indonesian using a trained model.
- **Modular Code**: Organized into separate modules for data handling, model definition, training, and inference.
- **Configurable**: Uses a YAML configuration file to manage hyperparameters and settings.

## Prerequisites

### Dependencies
The project requires Python 3.8 or higher. Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

The requirements.txt file includes:
```bash
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
tqdm>=4.62.0
pyyaml>=6.0.0
```

## Hardware
A GPU is recommended for faster training, but the code can run on a CPU as well.
Minimum 8GB RAM for small datasets.

