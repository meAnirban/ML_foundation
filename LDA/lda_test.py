import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from lda import LDA

data = datasets.load_iris()
X = data.data
y = data.target

lda = LDA(2)
lda.fit(X, y)
X_projected = lda.transform(X)

print('Shape of X', X.shape)
print('shape of transformed', X_projected.shape)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

plt.scatter(x1, x2, c = y, edgecolors='none', alpha=0.8, cmap = plt.cm.get_cmap('viridis', 3))
plt.xlabel('Principal component 1')
plt.ylabel("principal component 2")
plt.colorbar()
plt.show()