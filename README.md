# Bigram Pixel Language Model for MNIST

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Dataset: MNIST](https://img.shields.io/badge/Dataset-MNIST-blue.svg)](http://yann.lecun.com/exdb/mnist/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A minimal PyTorch implementation of a bigram language model operating at the pixel level, adapted for MNIST handwritten digit generation. Part of foundational AI/ML studies aligned with long-term deep learning mastery goals.

## Key Features

🖼️ **Pixel-level Generation** - Predicts next pixel intensity using bigram statistics  
🔢 **21-Class Classification** - Discretizes pixel values into 0.05 increments (0.0-1.0 range)  
📈 **Transformer Fundamentals** - Implements core language modeling concepts on image data  
🧮 **MNIST Adaptation** - Applies text-style modeling to 28x28 grayscale images  

## Implementation Overview

### Problem Transformation
```python
# Convert continuous pixel values [0,1] to 21 discrete classes
def quantize_pixels(image):
    return (image * 20).round().long()

# Example pixel value mapping:
# 0.00 → class 0, 0.05 → class 1, ..., 1.00 → class 20