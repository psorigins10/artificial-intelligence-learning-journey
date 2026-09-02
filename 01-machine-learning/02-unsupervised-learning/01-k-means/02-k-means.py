from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = [
    [1, 1],
    [2, 1],
    [1, 2],
    [8, 8],
    [9, 8],
    [8, 9]
]

kmeans = KMeans(
    n_clusters=2,
    random_state=42
)

kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_

# Plot the points
plt.scatter(
    [point[0] for point in X],
    [point[1] for point in X],
    c=labels
)

# Plot the centroids
centroids = kmeans.cluster_centers_

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=200
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("K-Means Clustering")

plt.show()