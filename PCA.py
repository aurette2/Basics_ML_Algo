import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self,n_component):
        self.nComponent = n_component
        self.eigen_vectors_sorted = None
        self.eigen_values_sorted = None

    def mean(self, X):
        return np.sum(X, axis=0) / X.shape[0]

    def std(self, X):
        return np.sqrt(np.sum((X - self.mean(X)) ** 2, axis=0) / (X.shape[0] - 1))

    def Standardize_data(self, X):
        return (X - self.mean(X)) / self.std(X)

    def covariance(self,X):
        return np.dot(np.transpose(X), X) / (X.shape[0] - 1)

    def fit(self, X):
        eigen_values, eigen_vectors = np.linalg.eig(self.covariance(self.Standardize_data(X)))
        idx = np.array([np.abs(i) for i in eigen_values]).argsort()[::-1]
        self.eigen_values_sorted = eigen_values[idx]
        self.eigen_vectors_sorted = eigen_vectors.T[:,idx]
        explained_variance = np.round([(i / sum(eigen_values))*100 for i in self.eigen_values_sorted],2)
        cum_explained_variance = np.cumsum(explained_variance)
        print(f"Explained_variance: {explained_variance}")
        print(f"cum_explained_variance: {cum_explained_variance}")

    def transform(self, X):
        """Get our projection matrix"""
        P = self.eigen_vectors_sorted[:self.nComponent, :] # Projection matrix
        return X.dot(P.T)
    
    def plot(self,X,y):
        plt.title(f"PC1 vs PC2")
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.xlabel('PC1'); plt.xticks([])
        plt.ylabel('PC2'); plt.yticks([])