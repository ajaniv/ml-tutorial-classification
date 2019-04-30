"""
It is a type of unsupervised algorithm which deals with the
clustering problems. Its procedure follows a simple and easy way
to classify a given data set through a certain number of
clusters (assume k clusters). Data points inside a cluster
are homogeneous and are heterogeneous to peer groups.
"""
# pylint: disable=invalid-name
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans

style.use("ggplot")

x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]

plt.scatter(x, y)
plt.show()
X = np.array([[1, 2],
              [5, 8],
              [1.5, 1.8],
              [8, 8],
              [1, 0.6],
              [9, 11]])
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print(centroids)
print(labels)
colors = ["g.", "r.", "c.", "y."]

for i in range(len(X)): # pylint: disable=consider-using-enumerate
    print("coordinate:", X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker="x", s=150, linewidths=5, zorder=10)
plt.show()

"""
[[1.16666667 1.46666667]
 [7.33333333 9.        ]]
[0 1 0 1 0 1]
coordinate: [1. 2.] label: 0
coordinate: [5. 8.] label: 1
coordinate: [1.5 1.8] label: 0
coordinate: [8. 8.] label: 1
coordinate: [1.  0.6] label: 0
coordinate: [ 9. 11.] label: 1
""" # pylint: disable=pointless-string-statement
