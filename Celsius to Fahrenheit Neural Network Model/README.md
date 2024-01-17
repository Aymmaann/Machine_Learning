# Celsius to Fahrenheit Neural Network Model

## Overview
This project implements a neural network model using TensorFlow and Keras to predict Fahrenheit temperatures given Celsius temperatures. The model is trained on a small dataset of Celsius and Fahrenheit temperature pairs. The trained model can then be used to predict Fahrenheit temperatures for new Celsius values.

## Dataset
The dataset used for training consists of a few Celsius and their corresponding Fahrenheit values:

```python
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46.4, 59, 71.6, 100.4], dtype=float)
```

## Code Structure
- The neural network model is defined using the Sequential API from TensorFlow and Keras.
- The model is compiled using the Adam optimizer with a learning rate of 0.1 and the mean squared error loss function.
- The training data (Celsius and Fahrenheit arrays) are then used to train the model for 500 epochs.
- The trained model is used to predict the Fahrenheit temperature for a new Celsius value (e.g., 100.0).
- The training loss over epochs is plotted using Matplotlib.

## Code Usage
1. Install the required libraries:
```bash
pip install tensorflow numpy matplotlib
```

2. Run the script:
```bash
python celsius_to_fahrenheit_predictor.ipynb
```

## Results
The model's predictions can be observed by running the script. Additionally, a plot of the training loss over epochs is displayed to visualize the learning progress.

*Feel free to experiment with the dataset, model architecture, or training parameters to enhance the model's performance.*