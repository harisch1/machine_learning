import numpy as np
import matplotlib.pyplot as plt


def ordinal_encode(data: np.array):
    """ordinal encoder for a numpy array feature by feature."""
    encoded = []
    for i in range(data.shape[1]):
        unique_values = np.unique(data[:, i])
        encoding = {value: i for i, value in enumerate(unique_values)}
        encoded.append(np.array([encoding[value] for value in data[:, i]]).reshape(-1, 1))
    return np.concatenate(encoded, axis=1)


def plot_kmeans(X, y, kmeans):
    # Plotting data
    plt.scatter(X[:, 0], X[:, 1], c=y)
    # Plotting centroids
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker="x", s=200)
    plt.show()

# TODO
def f1_score(y, y_hat):
    raise NotImplementedError("Implement the f1_score function")


def roc_auc_score(y, y_hat):
    raise NotImplementedError("Implement the roc_auc_score function")

def accuracy(y, y_hat):
    return np.sum(y == y_hat) / len(y)