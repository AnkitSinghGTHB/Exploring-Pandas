import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import random

data = pd.read_csv("dataset.csv")
print(data.describe())
print(data.info())
print(data.columns)
print(data.corr().to_string())

X=data[['age', 'sex', 'cp', 'trestbps', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
Y=data['target']
print(X)
print(Y)
sns.scatterplot(data=data)
plt.plot(X,Y)
plt.show()

#Linear Regression
X=data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
Y=data['target']
random.seed(1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.30)
regr = LinearRegression()
regr.fit(X_train, Y_train)
Y_pred = regr.predict(X_test)
Y_pred_rounded = np.round(Y_pred)

accuracy = accuracy_score(Y_test, Y_pred_rounded)
print(f"Accuracy Score: {accuracy * 100:.2f}%")

#Kth Nearest Neighbouur
X=data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
Y=data['target']
random.seed(1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.30)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, Y_train)


Y_pred = model.predict(X_test)
Y_pred_rounded = np.round(Y_pred)

accuracy = accuracy_score(Y_test, Y_pred_rounded)
print(f"Accuracy Score: {accuracy * 100:.2f}%")

#Logistic Regression
X=data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
Y=data['target']
random.seed(1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.30)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)


Y_pred = model.predict(X_test)
Y_pred_rounded = np.round(Y_pred)

accuracy = accuracy_score(Y_test, Y_pred_rounded)
print(f"Accuracy Score: {accuracy * 100:.2f}%")
