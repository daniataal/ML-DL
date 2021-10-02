'''
@Author: Dani Atalla
Date: 09/09/21
Time:12:40
'''

import numpy as np
from numpy.random import randint
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
data_plot = np.c_[X,y]


train_idx = randint(0, 150, 100)
X_train = X[train_idx]
y_train = y[train_idx]

test_idx = randint(0, 150, 50)
X_test = X[test_idx]
y_test = y[test_idx]




def eucledian(p1, p2):
    dist = np.sqrt(np.sum((p1 - p2) ** 2))
    return dist

def predict(x_train, y, x_input, k):
    op_labels = []

    # Loop through the Datapoints to be classified
    for item in x_input:

        # Array to store distances
        point_dist = []
        weights = []
        # Loop through each training Data
        for j in range(len(x_train)):
            distance = eucledian(np.array(x_train[j, :]), item) + 0.001
            weight = 1 / np.exp(distance)
            # Calculating the distance
            point_dist.append(distance)
            weights.append(weight)
        point_dist = np.array(point_dist)
        weights = np.array(weights)


        # Sorting the array while preserving the index
        # Keeping the first K datapoints
        dist = np.argsort(point_dist)[:k]

        # Labels of the K datapoints from above
        labels = y[dist]
        weights = weights[dist]

        # Majority voting
        # count = np.bincount(labels) #Count number of occurrences of each value in array of non-negative ints.
        # op_labels.append(np.argmax(count))
        # print("Point:", item, "Lable:", np.argmax(count))
        weighted_freq = []
        for label in np.unique(y):
            sm = 0
            for i in labels:
                if label == labels[i]:
                    sm += weights[i]
            weighted_freq.append(sm)
        prediction = np.argmax(weighted_freq)
        op_labels.append(prediction)
    return op_labels

# Checking the accuracy
def accuracy_score(y_test, y_pred):
    cor = 0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            cor += 1
    return cor / float(len(y_test)) *100.0


# Applying our function
# if __name__ == '__main':
#     y_pred = predict(X_train, y_train, X_test, 7)
#     print("Accuracy: ", accuracy_score(y_test, y_pred),"%")
y_pred = predict(X_train, y_train, X_test, 6)
print("Accuracy: ", accuracy_score(y_test, y_pred),"%")