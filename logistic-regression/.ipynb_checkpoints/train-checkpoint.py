import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from logisticRegression import LogisticRegression

bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target
x_train, x_test, y_train, y_test = model_selection(x, y, test_size=0.2, random_state=1234)
classifier = LogisticRegression()
classifier.fit(x_train, y_train)
pred = classifier.predict(x_test)

def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)
acc = accuracy(pred, y_test)
print(acc)


