# bigramLanguageModel

A minimal implementation of a bigram character-level language model, designed as part of a foundational study into transformer architectures and language modeling. This project serves as a learning tool for understanding core concepts in deep learning and AI, aligned with a long-term goal of mastering machine learning fundamentals.

## Overview

The model generates text by predicting the next character based on bigram (two consecutive characters) statistics. Two approaches are explored:
1. **Statistical Approach**: Directly computes bigram probabilities using frequency counts and normalization.
2. **Neural Network Approach**: Replicates the same task using a simple PyTorch neural network, demonstrating how gradient-based optimization achieves similar results.

Both approaches yield comparable performance, highlighting the relationship between statistical methods and neural network training.

## Code Structure

- Jupyter notebooks containing implementations: 
  - **Bigram Frequency Analysis**: `bigram.ipynb` Probability-based approach using tensors.
  - **Neural Network Implementation**: `bigram-nn.ipynb` PyTorch-based model trained with negative log-likelihood loss.

## Installation

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate  # Windows