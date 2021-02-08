import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=[200, 200, 100, 100], n_features=2,
                    centers=[[1, 1], [1,5],[5,5],[5,1]],
                    cluster_std=2)
plt.figure(figsize=(10, 10))

# plot ground true
plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Ground True")


for s in range(222,225,1):
    # using k-means method and random initialization to clustering
    y1 = KMeans(init='random', n_clusters=4).fit_predict(X)

    # plot the clustering result
    plt.subplot(s)
    plt.scatter(X[:, 0], X[:, 1], c=y1)
    plt.title("K-means cluster"+str(s-221))

    # compute the sum of square error
    squareError = 0
    for c in range(4):
        loc = np.where(y1 == c)
        cluster = X[loc]
        squareError += np.sum(np.square(cluster - cluster.mean(axis=0)))
    print("The square error after k-means" + str(s-221) +" is " + str(squareError))

squareError_true = 0
for c in range(4):
    loc = np.where(y == c)
    cluster = X[loc]
    squareError_true += np.sum(np.square(cluster - cluster.mean(axis=0)))
print("The true square error is " + str(squareError_true))
plt.show()
