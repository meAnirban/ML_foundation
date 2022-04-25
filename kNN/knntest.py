import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap = ListedColormap(['blue', 'red', 'green'])

iris = datasets.load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#print(X_train.shape)

#print(y_train.shape)
#print(y_train)


#plt.figure()
#plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k', s=20)
#plt.show()

#a = [1, 2, 3,1,6,3,6,7,9,1,1]
#from collections import Counter
#most_comm = Counter(a).most_common(2)
#print(most_comm)

from knn import KNN

model = KNN(k = 3)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(pred)

acc = np.sum(pred == y_test) / len(y_test)

print(acc)
