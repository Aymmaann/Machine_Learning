# Breast Cancer Classification with TensorFlow

This repository contains a simple TensorFlow script for building and training a neural network to classify breast cancer using the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn.

## Prerequisites

Make sure you have the required dependencies installed:

- TensorFlow
- scikit-learn
- Matplotlib

You can install them using the following:

```bash
pip install tensorflow scikit-learn matplotlib
```

## Dataset
The Breast Cancer Wisconsin (Diagnostic) dataset is loaded using scikit-learn's *load_breast_cancer* function. The dataset consists of features related to breast cancer cell characteristics, and the goal is to classify tumors as malignant or benign.

## Data Preprocessing
- The dataset is split into training and testing sets using train_test_split from scikit-learn.
- Standard scaling is applied to normalize the feature values using StandardScaler from scikit-learn.

## Neural Network Architecture
A simple neural network with one dense layer is created using TensorFlow's Keras API. The model uses the sigmoid activation function for binary classification.

```Python
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(D,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

## Model Compilation
The model is compiled with the Adam optimizer and binary cross-entropy loss for binary classification. The accuracy metric is used for evaluation.

```Python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

## Model Training
The model is trained on the training set and validated on the testing set for 100 epochs.

```Python
r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)
```

## Model Evaluation
The training and testing performance is visualized using Matplotlib. Additionally, the script prints the final evaluation scores on both the training and testing sets.

## Usage
Run the script in a notebook and observe the training process and final evaluation scores.

Feel free to experiment with hyperparameters, model architecture, or explore more advanced techniques for improved performance.