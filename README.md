# Fruit and Vegetable Classification

This repository contains a machine learning project for classifying healthy and rotten fruits and vegetables using a convolutional neural network (CNN). The dataset consists of labeled images and the model employs data augmentation and batch normalization for improved accuracy.

## Dataset

The dataset used in this project contains images of various fruits and vegetables, categorized into healthy and rotten classes. The images have been resized to 224x224 pixels for consistency.

**Note:** The raw dataset is not included in this repository due to its large size. You can download the dataset from the following link:
[Download Dataset](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten)

## Project Structure

├── Raw Data
│   └── Fruit And Vegetable Diseases Dataset
│       ├── Apple__Healthy
│       ├── Apple__Rotten
│       ├── …
├── src
│   ├── data_loading.py
│   ├── model_training.py
│   ├── model_evaluation.py
├── README.md
└── requirements.txt

## Data Augmentation

The project uses the `ImageDataGenerator` class from Keras for data augmentation, including:

- Rotation
- Width shift
- Height shift
- Shear
- Zoom
- Horizontal flip

## Model Architecture

The model is built using Keras' Sequential API and includes:

- Convolutional layers with ReLU activation
- MaxPooling layers
- Batch Normalization
- Dense layers with Dropout for regularization
- Sigmoid activation for binary classification

## Training

The model is trained with the following parameters:

- Batch size: 32
- Epochs: 20
- Optimizer: Adam with a learning rate of 0.0001
- Loss function: Binary Crossentropy

## Evaluation

The model is evaluated on a separate test set to determine its accuracy. The project includes scripts for visualizing the training process and evaluating the model's performance.

## Usage

1. Clone the repository:
   ```sh
   git clone https://github.com/JustJin28/Fruit-Ripeness-Prediction.git
   cd fruit-vegetable-classification
