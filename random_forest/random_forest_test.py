import numpy as np
from sklearn import datasets
from random_forest import RandomForest
from sklearn.model_selection import train_test_split

data = datasets.load_breast_cancer()

X  = data.data
y = data.target

def accuracy(y_test, y_pred):
    return np.sum( y_test == y_pred) / len(y_test)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

clf = RandomForest(n_trees = 10)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(accuracy(y_test, y_pred))