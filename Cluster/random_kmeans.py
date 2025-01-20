import torch
import numpy as np

def kmeans(X, k, max_iters=100, tol=1e-4):
    """
    Using PyTorch implement K-means algorithm
    
    參數:
    X: torch.Tensor, shape as (n_samples, n_features)
    k: int, number of cluster
    max_iters: int, maximum iteration
    tol: float, threshold
    
    return:
    centroids: torch.Tensor, shape as (k, n_features)
    labels: torch.Tensor, shape as (n_samples,)
    """
    
    # Randomly choose k data point as initial center
    n_samples = X.shape[0]
    centroid_indices = torch.randperm(n_samples)[:k]
    centroids = X[centroid_indices]
    
    for _ in range(max_iters):
        # Calculate the distance for each data point to central
        distances = torch.cdist(X, centroids)
        
        # Arrange each data point to most shorest central
        labels = torch.argmin(distances, dim=1)
        
        # Update central
        new_centroids = torch.stack([X[labels == i].mean(dim=0) for i in range(k)])
        
        # Check whether convergence
        if torch.all(torch.abs(new_centroids - centroids) < tol):
            break
        
        centroids = new_centroids
    
    return centroids, labels

# Example
if __name__ == "__main__":
    # Generate randomly data
    np.random.seed(0)
    X = np.random.randn(1000, 2)
    X = torch.tensor(X, dtype=torch.float32)

    # Execute K-means
    k = 3
    centroids, labels = kmeans(X, k)

    print("Centroids:")
    print(centroids)
    print("\nFirst few labels:")
    print(labels[:10])

    # Matplot display the clustering result
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X')
    plt.title('K-means Clustering Result')
    plt.show()
    plt.savefig("random_kmeans.png")
