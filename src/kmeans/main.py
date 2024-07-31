import numpy as np


class KMeans:
    def __init__(self, k: int):
        self.k = k
        self.centroids: np.array

    @staticmethod
    def distance(point: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """returns the distance between a given data point and all centroids."""
        return np.sqrt(np.sum((centroids - point) ** 2, axis=1))

    def fit(self, X: np.ndarray, max_iter: int=200, delta=0.0001):
        # Initialize random centroids within the min and max of X
        # Centroid will have the same shape as a data point in X
        self.centroids = np.random.uniform(
            np.amin(X, axis=0), np.amax(X, axis=0), (self.k, X.shape[1])
        )

        for _ in range(max_iter):
            # Index of the label repsents the datapoint and the value represents the centroid
            labels = []
            for x in X:
                # Calculate the distance for each datapoint, get the index of the minimum distance and append it to the labels
                labels.append(np.argmin(self.distance(x, self.centroids)))

            # Update centroids
            new_centroids = []
            for i in range(self.k):
                # Get the data points that belong to a certain cluster
                data_points = X[np.argwhere(labels == i)]
                # Update the centroid
                if len(data_points) > 0:
                    # Get the mean of the data points to set the updated centroid
                    new_centroids.append(np.mean(data_points))
                else:
                    # If there are no data points in the cluster, assign the same value to the centroid
                    new_centroids.append(self.centroids[i])
    
            if np.max(np.array(new_centroids) - self.centroids) < delta:
                break
            self.centroids = np.array(new_centroids)
        
        return labels
    
    def predict(self, X: np.ndarray):
        labels = []
        for x in X:
            labels.append(np.argmin(self.distance(x, self.centroids)))
        return labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X = np.random.randint(0, 100, (100, 2))
    kmeans = KMeans(5)
    labels = kmeans.fit(X, 10000)
    # print(labels)

    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c=range(kmeans.k), marker="x", s=200)
    plt.show()
