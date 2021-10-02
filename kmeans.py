'''
Author:Dani Atalla
Date:09/09/21
Time:14:40
'''


import numpy as np
import matplotlib.pyplot as plt

X = -0.5 + np.random.rand (100, 2)
X1 = 0.5 + np.random.rand (50, 2)
X[50:100, :] = X1
plt.scatter (X[:, 0], X[:, 1], s=20, c='k')

centroids = np.random.rand (2, 2)
centroids

"""Let's visualize the dataset and the centroids in the same plot."""

plt.scatter (X[:, 0], X[:, 1], s=20, c='k')
plt.scatter (centroids[:, 0], centroids[:, 1], s=50, c='b', marker='+')

"""Using the function `np.linalg.norm()` from numpy we can calculate the Euclidean distance from each point to each centroid.
For instance, the following code is used to calculate the distances from all the points stored in the variable $X$ to the first centroid.
Then we print the first 10 distances."""

dist = np.linalg.norm (X - centroids[0, :], axis=1).reshape (-1, 1)
dist[:10, :]

"""Now we add the distance from all the points to the second centroid to the variable `dist` defined above.
This will give as a matrix with N rows and 2 columns, where each row refers to one point of $X$,
and each column is the distance value from one point to one of the centroids."""

dist = np.append (dist, np.linalg.norm (X - centroids[1, :], axis=1).reshape (-1, 1), axis=1)
dist[:10, :]


classes = np.argmin (dist, axis=1)
classes

"""Visualize how the points are being currently classified."""

plt.scatter (X[classes == 0, 0], X[classes == 0, 1], s=20, c='b')
plt.scatter (X[classes == 1, 0], X[classes == 1, 1], s=20, c='r')

"""Now we update the position of each centroid, by calculating it at the mean position of the cluster. For instance, if a certain point has the points (1,0), (2,1) and (0.5,0.5), then the updated position of the centroid is:

$$
c_j = ((1 + 2 + 0.5)/3, (0 + 1 + 0.5)/3)
$$
"""

# update position
for class_ in set (classes):
    centroids[class_, :] = np.mean (X[classes == class_, :], axis=0)
centroids

"""To understand what is happening here, let's visualize the dataset with the updated positioning of the centroids."""

plt.scatter (X[classes == 0, 0], X[classes == 0, 1], s=20, c='b')
plt.scatter (X[classes == 1, 0], X[classes == 1, 1], s=20, c='r')
plt.scatter (centroids[:, 0], centroids[:, 1], s=50, c='k', marker='+')


class KMeans:
    def __init__(self, k):
        self.k = k

    def train(self, X, MAXITER=100, TOL=1e-3):
        centroids = np.random.rand (self.k, X.shape[1])
        centroidsold = centroids.copy ()
        for iter_ in range (MAXITER):
            dist = np.linalg.norm (X - centroids[0, :], axis=1).reshape (-1, 1)
            for class_ in range (1, self.k):
                dist = np.append (dist, np.linalg.norm (X - centroids[class_, :], axis=1).reshape (-1, 1), axis=1)
            classes = np.argmin (dist, axis=1)
            # update position
            for class_ in set (classes):
                centroids[class_, :] = np.mean (X[classes == class_, :], axis=0)
            if np.linalg.norm (centroids - centroidsold) < TOL:
                break
                print ('Centroid converged')
        self.centroids = centroids

    def predict(self, X):
        dist = np.linalg.norm (X - self.centroids[0, :], axis=1).reshape (-1, 1)
        for class_ in range (1, self.k):
            dist = np.append (dist, np.linalg.norm (X - self.centroids[class_, :], axis=1).reshape (-1, 1), axis=1)
        classes = np.argmin (dist, axis=1)
        return classes


"""Let's test our class by defining a KMeans classified with two centroids (k=2) and training in dataset $X$, as it was done step-by-step above."""

kmeans = KMeans (2)
kmeans.train (X)


classes = kmeans.predict (X)
classes


plt.scatter (X[classes == 0, 0], X[classes == 0, 1], s=20, c='b')
plt.scatter (X[classes == 1, 0], X[classes == 1, 1], s=20, c='r')
plt.scatter (kmeans.centroids[:, 0], kmeans.centroids[:, 1], s=50, c='k', marker='+')


X = -0.5 + np.random.rand (100, 3)
X1 = 0.5 + np.random.rand (33, 3)
X2 = 2 + np.random.rand (33, 3)
X[33:66, :] = X1
X[67:, :] = X2

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure (figsize=(8, 5))
ax = fig.add_subplot (111, projection='3d')
ax.scatter (X[:, 0], X[:, 1], X[:, 2])

kmeans = KMeans (3)
kmeans.train (X)

kmeans.centroids

classes = kmeans.predict (X)
classes

fig = plt.figure (figsize=(8, 5))
ax = fig.add_subplot (111, projection='3d')
ax.scatter (X[classes == 0, 0], X[classes == 0, 1], X[classes == 0, 2])
ax.scatter (X[classes == 1, 0], X[classes == 1, 1], X[classes == 1, 2])
ax.scatter (X[classes == 2, 0], X[classes == 2, 1], X[classes == 2, 2])

