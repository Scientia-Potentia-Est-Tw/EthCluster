import torch
import numpy as np

def kmeans(X, k, max_iters=100, tol=1e-4):
    """
    使用PyTorch實現K-means算法
    
    參數:
    X: torch.Tensor, 形狀為 (n_samples, n_features)
    k: int, 聚類的數量
    max_iters: int, 最大迭代次數
    tol: float, 收斂閾值
    
    返回:
    centroids: torch.Tensor, 形狀為 (k, n_features)
    labels: torch.Tensor, 形狀為 (n_samples,)
    """
    
    # 選擇k個隨機數據點作為初始中心
    n_samples = X.shape[0]
    centroid_indices = torch.randperm(n_samples)[:k]
    centroids = X[centroid_indices]
    
    for _ in range(max_iters):
        # 計算每個數據點到每個中心的距離
        distances = torch.cdist(X, centroids)
        
        # 為每個數據點分配最近的中心
        labels = torch.argmin(distances, dim=1)
        
        # 更新中心
        new_centroids = torch.stack([X[labels == i].mean(dim=0) for i in range(k)])
        
        # 檢查是否收斂
        if torch.all(torch.abs(new_centroids - centroids) < tol):
            break
        
        centroids = new_centroids
    
    return centroids, labels

# 使用示例
if __name__ == "__main__":
    # 生成隨機數據
    np.random.seed(0)
    X = np.random.randn(1000, 2)
    X = torch.tensor(X, dtype=torch.float32)

    # 運行K-means
    k = 3
    centroids, labels = kmeans(X, k)

    print("Centroids:")
    print(centroids)
    print("\nFirst few labels:")
    print(labels[:10])

    # 可視化結果（如果需要）
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X')
    plt.title('K-means Clustering Result')
    plt.show()
    plt.savefig("random_kmeans.png")