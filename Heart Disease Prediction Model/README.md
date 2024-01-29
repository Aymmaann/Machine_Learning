# Heart Disease Prediction Model

## Overview

This repository contains a simple TensorFlow model for predicting heart disease based on the Heart Disease UCI dataset. The model is a binary classification model that predicts the likelihood of heart disease.

## Dataset

The dataset used in this project is the Heart Disease UCI dataset, which is available in the file `heart.csv`. It contains various features related to heart health, and the goal is to predict the presence or absence of heart disease.

## Dependencies

- TensorFlow 2.x
- Numpy
- Pandas
- Matplotlib
- Scikit-learn

Install the required dependencies using:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

Certainly! Below is the README content in Markdown format:

markdown
Copy code
# Heart Disease Prediction Model

## Overview

This repository contains a simple TensorFlow model for predicting heart disease based on the Heart Disease UCI dataset. The model is a binary classification model that predicts the likelihood of heart disease.

## Dataset

The dataset used in this project is the Heart Disease UCI dataset, which is available in the file `heart.csv`. It contains various features related to heart health, and the goal is to predict the presence or absence of heart disease.

## Dependencies

- TensorFlow 2.x
- Numpy
- Pandas
- Matplotlib
- Scikit-learn

Install the required dependencies using:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

## Exploratory Data Analysis
The project includes an exploratory data analysis (EDA) step to visualize the relationships between selected features using scatter plots. The EDA is implemented using pandas and matplotlib.

## Model Architecture
The model architecture consists of a single dense layer with a sigmoid activation function, suitable for binary classification.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(D,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

## Model Training
The model is trained using the Adam optimizer with a custom learning rate schedule. The training history is plotted to visualize accuracy improvements over epochs.
```python
scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)
r = model.fit(X_train, y_train, validation_data=[X_test, y_test], epochs=100, callbacks=[scheduler])
```

## Model Evaluation
After training, the model's performance is evaluated on both the training and test sets.
```python
print(f"Train Score: {model.evaluate(X_train, y_train)}")
print(f"Test Score: {model.evaluate(X_test, y_test)}")
```

## Custom Learning Rate Schedule
A custom learning rate schedule is implemented to adjust the learning rate during training.
```python
def schedule(epochs, lr):
    if epochs >= 50:
        return 0.001
    return 0.01

scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)
```

## Customization

Feel free to customize and extend this code for your specific use case. You have the flexibility to tune the model with different hyperparameters, hidden layers, and callback functions. The provided architecture, hyperparameters, and callbacks are chosen based on experimentation and have shown promising results during testing.

Experiment with the following aspects for potential improvements:

- **Model Architecture:** Adjust the number of layers, neurons, or activation functions to suit your data and problem.

- **Hyperparameters:** Fine-tune parameters like learning rate, regularization strength, and batch size to optimize model performance.

- **Callback Functions:** Explore different callback functions for training, such as early stopping, model checkpointing, or custom learning rate schedules.

Remember that the effectiveness of these adjustments may vary based on your dataset and problem characteristics. The provided settings serve as a starting point, and you are encouraged to iterate and find the configuration that best suits your requirements.
