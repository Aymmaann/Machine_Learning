#Importing modules
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

#Loading the dataset
columns=['SepalLength(cm)', 'SepalWidth(cm)', 'PetalLength(cm)',␣ 'PetalWidth(cm)', 'Class']
df = pd.read_csv( 'iris.csv', names = columns)
df.head()

#Analyzing the dataset
df.describe()
df.info()

#Visualizing the dataset
sns.pairplot(df, hue='Class')

#Evaluating the performance of the model
from sklearn.model_selection import train_test_split

#train: 80% and test: 20%
X=df.drop(columns = ['Class'])
Y=df['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)
LogisticRegression()
print("Accuracy of model :", model.score(X_test, Y_test)*100)

#We can test the model using a few other algorithms too, a few are given below

#Using Support Vector Machine
# train: 70% and test: 30%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, Y_train)
SVC()
print("Accuracy of model :", model.score(X_test, Y_test)*100)

#Using K-Nearest Neighbour
# train: 70% and test: 30%
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
KNeighborsClassifier()
print("Accuracy of the model :", model.score(X_test, Y_test)*100)

#Using Decision tree
# train: 70% and test: 30%
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
DecisionTreeClassifier()
print("Accuracy of the model :", model.score(X_test, Y_test)*100)

#Using Naive Bayes Classifier
# train: 70% and test: 30%
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, Y_train)
GaussianNB()
print("Accuracy of the model :", model.score(X_test, Y_test)*100) 
