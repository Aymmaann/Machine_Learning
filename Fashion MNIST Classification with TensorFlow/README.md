# Fashion MNIST Classification with TensorFlow

This repository contains a simple TensorFlow script for training and evaluating a neural network to classify images from the Fashion MNIST dataset.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Callback](#callback)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)

## Introduction

The script uses TensorFlow and Keras to build a neural network for image classification. The model is trained on the Fashion MNIST dataset, a collection of grayscale images of clothing items.

## Dependencies

- TensorFlow
- NumPy
- Matplotlib

Install the required dependencies using:

```bash
pip install tensorflow numpy matplotlib
```

## Usage

To run the script, execute the following command:
```bash
python fashion.ipynb
```

## Callback
The script uses a custom callback (CallBack) to stop training once the accuracy reaches 95%. This prevents unnecessary computation if the desired accuracy is achieved early.

## Dataset
The Fashion MNIST dataset is used for training and testing. It is loaded using TensorFlow's built-in dataset loader.


## Model Architecture
The neural network architecture is defined as follows:
```bash
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

## Training
The model is compiled using the Adam optimizer and sparse categorical crossentropy loss. It is then trained on the training dataset for 10 epochs.
```bash
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10)
```

## Evaluation
The trained model is evaluated on the test dataset using:
```bash
model.evaluate(testing_images, testing_labels)
```

## Results
The script prints the predicted class probabilities and the actual label for the first test image.
```bash
classification = model.predict(testing_images)
print(classification[0])
print(testing_labels[0])
```

*Feel free to explore the code and adapt it to your specific needs. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.*
