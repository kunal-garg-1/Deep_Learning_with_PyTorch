# PyTorch Neural Network Classification 🧠

This repository contains a comprehensive Google Collab Notebooks dedicated to understanding and implementing **Neural Network Classification** using PyTorch. 

Whether classifying data into two categories (binary) or multiple categories (multi-class), this project breaks down the entire deep learning pipeline—from dataset generation to model evaluation and visualization.

## 📌 Overview

Classification is one of the most foundational problems in machine learning. This notebook demonstrates how to build neural network classification models from scratch, exploring the limitations of strictly linear models and showing how non-linear activation functions help neural networks learn highly complex patterns.

### Key Concepts Covered:
- **Neural Network Architecture:** Understanding input/output shapes, hidden layers, and how to construct a model by subclassing `nn.Module`.
- **Binary Classification:** - Generating a synthetic 2D dataset (e.g., using scikit-learn's `make_circles`).
  - Building a PyTorch model to classify two distinct categories.
  - Applying `BCEWithLogitsLoss` (Binary Cross Entropy).
- **Multi-Class Classification:# PyTorch Deep Learning & Machine Learning Journey 🧠

Welcome to my central repository for learning **PyTorch**! This repository contains all the code, notes, and exercises I am writing as I progress through the incredible [PyTorch for Deep Learning & Machine Learning – Full Course](https://www.youtube.com/watch?v=V_xro1bcAuA) by Daniel Bourke (via freeCodeCamp).

## 📌 Repository Overview
This repository documents my progression from the absolute fundamentals of PyTorch tensors to building custom Convolutional Neural Networks (CNNs) for computer vision tasks using real-world custom datasets.

*Note: I am actively updating this repository as I learn and complete new parts of the course!*

---

## 🛠️ Skills & Concepts Covered

### Part 1: PyTorch Fundamentals
* **Tensor Basics:** Creating, shaping, and manipulating PyTorch tensors (the fundamental building block of machine learning).
* **Tensor Operations:** Matrix multiplication, indexing, reshaping, stacking, squeezing/unsqueezing, and tensor aggregation (min, max, mean).
* **PyTorch & NumPy:** Bridging PyTorch tensors and NumPy arrays.
* **Reproducibility:** Setting random seeds (`torch.manual_seed`) for reproducible experiments.
* **Device-Agnostic Code:** Writing hardware-agnostic code to run calculations seamlessly on CPUs or GPUs (CUDA).

### Part 2: PyTorch Workflow
* **End-to-End ML Pipeline:** Preparing data, building a model, fitting the model, making predictions, and evaluating.
* **Model Building:** Subclassing `torch.nn.Module` and utilizing `torch.nn.Linear` to build networks.
* **Training & Testing Loops:** Writing custom PyTorch optimization loops, computing loss, setting gradients to zero, and performing backpropagation (`loss.backward()`, `optimizer.step()`).
* **Saving & Loading Models:** Persisting trained models (`torch.save` and `load_state_dict`) for future inference.

### Part 3: Neural Network Classification
* **Binary & Multi-Class Classification:** Building architectures capable of classifying data into two or multiple categories.
* **Non-Linearities:** Understanding the power of non-linear activation functions (ReLU, Sigmoid, Softmax) to model complex, real-world data.
* **Loss Functions & Optimizers:** Utilizing `BCEWithLogitsLoss` and `CrossEntropyLoss` alongside SGD and Adam optimizers.
* **Model Evaluation:** Implementing accuracy metrics and visualizing model decision boundaries to see exactly what the model is learning.

### Part 4: Computer Vision & CNNs
* **Torchvision:** Utilizing the `torchvision` library (`datasets`, `transforms`, and `models`) for computer vision problems.
* **Convolutional Neural Networks (CNNs):** Understanding and implementing convolutional layers (`nn.Conv2d`) and pooling layers (`nn.MaxPool2d`).
* **Architecture Replication:** Building the classic **TinyVGG** architecture from scratch to classify image datasets (like FashionMNIST).
* **Visualizing Performance:** Evaluating models by generating and plotting Confusion Matrices using `torchmetrics` and `mlxtend`.

### Part 5: Custom Datasets
* **Data Loading:** Loading custom, real-world image datasets (e.g., Pizza, Steak, Sushi) using `torchvision.datasets.ImageFolder`.
* **Custom Dataset Classes:** Subclassing `torch.utils.data.Dataset` to build bespoke data loading pipelines from scratch.
* **Data Augmentation:** Artificially increasing dataset diversity and preventing overfitting using `torchvision.transforms` (e.g., `TrivialAugmentWide`).
* **Batches & DataLoaders:** Batching data using `torch.utils.data.DataLoader` for memory efficiency and faster GPU training.
* **Custom Inference:** Pre-processing raw, real-world images (resizing, changing data types, unsqueezing) and passing them through a trained model to make custom predictions.

---
### ⚙️ Technologies Used
* **Python 3.x**
* **PyTorch & Torchvision**
* **Matplotlib** (for data visualization and plotting loss curves)
* **Scikit-Learn** (for data splitting and toy datasets)
* **Pandas & NumPy**

---
*Created as part of an ongoing journey to master Deep Learning with PyTorch.***
  - Generating data with more than two classes (e.g., using `make_blobs`).
  - Adjusting the network's final layer to output multiple raw logits.
  - Applying `CrossEntropyLoss`.
- **The Power of Non-Linearity:** Implementing activation functions like ReLU (Rectified Linear Unit) alongside Sigmoid and Softmax to map non-linear data sets.
- **The PyTorch Training Loop:** Writing the standard PyTorch workflow (forward pass, calculating loss, zeroing gradients, backpropagation, and stepping the optimizer).
- **Evaluation & Visualization:** Writing helper functions to calculate accuracy and plotting decision boundaries to visually verify what the model has learned.

## 🛠️ Prerequisites & Setup

To run the notebook locally, you will need Python 3.x installed along with the following primary libraries. You can install them via pip:

```bash
pip install torch torchvision torchaudio
pip install matplotlib scikit-learn pandas numpy
