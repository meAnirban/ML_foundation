import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

bc = datasets.load_breast_cancer()

X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#print(X_train.shape)

#print(y_train.shape)
#print(y_train)


#plt.figure()
#plt.scatter(X[:, 0], y,  color='b', marker='o', s=30)
#plt.show()



from logistic_regression import LogisticRegression

model = LogisticRegression(lr = 0.0001, n_iters = 10000)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(pred)

acc = np.sum(pred == y_test) / len(y_test)

print(acc)


#cmap = plt.get_cmap('viridis')
#fig = plt.figure(figsize=(8,6))
#m1= plt.scatter(X_train, y_train, color = cmap(0.9), s= 10)
#m1= plt.scatter(X_test, y_test, color = cmap(0.5), s= 10)
#plt.plot(X, y_pred, color= 'black', linewidth =2, label = "Prediction")
#plt.show()


