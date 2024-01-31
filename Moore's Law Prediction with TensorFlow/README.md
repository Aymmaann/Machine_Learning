
# Moore's Law Prediction with TensorFlow
This script utilizes TensorFlow to create a simple neural network model to predict the doubling time of transistors according to Moore's Law. The data is loaded from the "moore.csv" file, and the model is trained to fit the exponential growth pattern.

## Dependencies
- TensorFlow
- NumPy
- Pandas
- Matplotlib

## Setup
Make sure to install the required dependencies:
```bash
pip install tensorflow numpy pandas matplotlib
```

## Usage
1. Clone the repository or download the "moore.csv" file.
```bash
!wget https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv
```
2. Run the script:

## Explanation
1. The data is loaded from "moore.csv" and preprocessed.
2. Logarithmic transformation is applied to the target variable to handle exponential growth.
3. The model is created using TensorFlow with a single dense layer.
4. Stochastic Gradient Descent (SGD) optimizer is used for training.
5. A learning rate scheduler is implemented to adjust the learning rate during training.
6. The model is trained for 200 epochs, and the loss is plotted.
7. The model's performance is evaluated on the training data.
8. The weights of the trained model are extracted to calculate the doubling time.

## Results
The script outputs the time taken for transistors to double according to Moore's Law based on the trained model which was 1.989 years in my case.

*Adjustments can be made to the model architecture, hyperparameters, and data preprocessing steps to improve performance*