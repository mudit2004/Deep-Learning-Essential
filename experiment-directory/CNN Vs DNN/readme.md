# DNN Vs CNN

This folder contains experiments and implementations of various deep learning models using the Fashion MNIST dataset. The aim is to explore and compare different neural network architectures and techniques.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Deep Neural Network (DNN)](#deep-neural-network-dnn)
- [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
- [Experiments with Learning Rates](#experiments-with-learning-rates)
- [Results](#results)
- [Future Work](#future-work)

## Introduction

This project explores the implementation and performance of different neural network architectures on the Fashion MNIST dataset. We start with a simple Deep Neural Network (DNN) and move on to a more complex Convolutional Neural Network (CNN). We also experiment with different learning rates to observe their effects on model performance.

## Dataset

The Fashion MNIST dataset is used for training and evaluating the models. It consists of 70,000 grayscale images in 10 categories. The images show individual articles of clothing at low resolution (28 by 28 pixels).

## Deep Neural Network (DNN)

We implemented a simple DNN with the following architecture:
- Flatten layer
- Dense layer with 100 neurons and ReLU activation
- Dense layer with 70 neurons and ReLU activation
- Dense layer with 50 neurons and ReLU activation
- Dense layer with 20 neurons and ReLU activation
- Dense layer with 10 neurons and Softmax activation

The DNN was compiled using Sparse Categorical Crossentropy as the loss function and Adam optimizer. The model was trained for 50 epochs, and the performance was evaluated on the test set.

### DNN Model Summary

```python
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(70, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```

## Convolutional Neural Network (CNN)

We implemented a CNN with the following architecture:
- Conv2D layer with 32 filters, kernel size (3, 3), and ReLU activation
- MaxPooling2D layer with pool size (2, 2)
- Conv2D layer with 64 filters, kernel size (3, 3), and ReLU activation
- MaxPooling2D layer with pool size (2, 2)
- Flatten layer
- Dense layer with 100 neurons and ReLU activation
- Dense layer with 70 neurons and ReLU activation
- Dense layer with 50 neurons and ReLU activation
- Dense layer with 20 neurons and ReLU activation
- Dense layer with 10 neurons and Softmax activation

The CNN was compiled using Sparse Categorical Crossentropy as the loss function and Adam optimizer. The model was trained with different learning rates, and the performance was evaluated on the test set.

### CNN Model Summary

```python
model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(70, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```

## Experiments with Learning Rates

We experimented with the following learning rates for the CNN model:
- 0.1
- 0.01
- 0.001
- 0.0001

For each learning rate, we trained the model for 10 epochs and recorded the training and validation loss and accuracy. We also calculated the accuracy, precision, recall, and F1 score on the test set.

### Training and Validation Loss

![Training and Validation Loss for Different Learning Rates](path/to/loss_plot.png)

### Training and Validation Accuracy

![Training and Validation Accuracy for Different Learning Rates](path/to/accuracy_plot.png)

## Results

The results of the experiments with different learning rates are summarized below:

| Learning Rate | Accuracy | Precision | Recall | F1 Score |
|---------------|----------|-----------|--------|----------|
| 0.1           | 0.1      | 0.01      | 0.1    | 0.0181   |
| 0.01          | 0.83     | 0.84      | 0.83   | 0.84     |
| 0.001         | 0.86     | 0.87      | 0.86   | 0.86     |
| 0.0001        | 0.80     | 0.81      | 0.80   | 0.80     |

## Future Work

In future experiments, we plan to:
- Apply a convolutional neural network for the same image classification dataset and compare DNN and CNN in terms of parameters and performance.
- Construct an object detector using a convolutional neural network.
- Develop an image segmentation model using a fully convolutional network.
- Demonstrate the use of an autoencoder for dimensionality reduction.
- Implement a deep reinforcement learning algorithm for dynamic prediction.

## License

This repository is licensed under the BSD-3 license.
