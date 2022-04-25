import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#print(X_train.shape)

#print(y_train.shape)
#print(y_train)


#plt.figure()
#plt.scatter(X[:, 0], y,  color='b', marker='o', s=30)
#plt.show()



from linear_regression import LinearRegression

model = LinearRegression(lr =0.01, n_iters = 500)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(pred)

rmse = np.sqrt(np.mean((pred - y_test)**2))

print(rmse)

y_pred = model.predict(X)

cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1= plt.scatter(X_train, y_train, color = cmap(0.9), s= 10)
m1= plt.scatter(X_test, y_test, color = cmap(0.5), s= 10)
plt.plot(X, y_pred, color= 'black', linewidth =2, label = "Prediction")
plt.show()


