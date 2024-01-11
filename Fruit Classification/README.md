# Fruit Classification using Machine Learning
This project involves the classification of fruits based on certain features using various machine learning algorithms. The dataset used in this project contains information about fruits, including features such as mass, width, height, and color score.

## Dataset
The dataset is stored in a CSV file named Fruit_data.csv and has the following columns:

- fruit_label: Numeric label for each fruit.
- fruit_name: Name of the fruit.
- fruit_subtype: Subtype or category of the fruit.
- mass: Mass of the fruit.
- width: Width of the fruit.
- height: Height of the fruit.
- color_score: Color score of the fruit.


## Code Overview
The code is structured as follows:

1. Data Loading and Exploration:
- The dataset is loaded using pandas from the CSV file.
- Initial exploration using head() and describe() functions.
- Checking unique values and group sizes for the target variable (fruit_name).
- Visualizing the distribution of fruit names using a count plot.


2. Data Preprocessing:
- Dropping the target variable (fruit_label) for exploratory data analysis (EDA).
- Applying label encoding to convert categorical variables (fruit_name and fruit_subtype) to numeric format.


3. Data Splitting:
- Splitting the dataset into training and testing sets using train_test_split.


4. Feature Scaling:
- Applying MinMaxScaler to scale the features within a specific range.


5. Model Training and Evaluation:
Implementing several classification algorithms:
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Support Vector Machine (SVM)

Training each model on the training set and evaluating on the test set.
Displaying accuracy scores for each model.


6. Additional Metrics:
 - Displaying a classification report for the K-Nearest Neighbors (KNN) algorithm, including precision, recall, and F1-score.

## Requirements
- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
- Run the code in a Jupyter notebook or a Python script.

*Feel free to modify and experiment with the code to enhance the performance of the models or try different algorithms.*

