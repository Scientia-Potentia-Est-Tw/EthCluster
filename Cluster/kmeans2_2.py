import os
import re
import torch
from gensim.models import Word2Vec, TfidfModel
from gensim.corpora import Dictionary
import pickle
import numpy as np
from kmeans_pytorch import kmeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load the first set of document vectors
#with open('../Contracts/access_control/document_vectors.pkl', 'rb') as f:
#    document_vectors1 = pickle.load(f)

# Load the second set of document vectors
#with open('../Contracts/clean/document_vectors.pkl', 'rb') as f:
#    document_vectors2 = pickle.load(f)


vulnerable = "access_control"
vuln_size = 18
running_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Word2Vec.load("../Contracts/vuln/" +  vulnerable + "/mix/word2vec_model.model")

random_seed = 1194
torch.manual_seed(random_seed)
np.random.seed(random_seed)
# Number of clusters - adjust based on your needs
num_clusters = 3
    
with open("../Contracts/vuln/" + vulnerable + "/mix/document_vectors.pkl", "rb") as f:
    document_vectors = pickle.load(f)

# Convert document vectors to PyTorch tensor
document_vectors_tensor = torch.tensor(document_vectors, dtype=torch.float32)

# Set the maximum number of iterations
max_iterations = 1000

# Perform K-means clustering
cluster_ids_x, cluster_centers = kmeans(
    X=document_vectors_tensor, 
    num_clusters=num_clusters, 
    distance='euclidean', 
    device = running_device,
    iter_limit=max_iterations,
)
torch.save(cluster_centers, "../Contracts/vuln/" + vulnerable + "/mix/cluster_centers")



sum = [0] * (num_clusters * 2)

# Print cluster assignments
print("\nCluster assignments:")
for i, cluster_id in enumerate(cluster_ids_x):
    print(f"Document {i + 1} assigned to cluster {cluster_id}")
    if i < vuln_size:
        sum[cluster_id] += 1
    else:
        sum[cluster_id + num_clusters] += 1

print(f"\n")

for i in range(num_clusters * 2):
    if i < num_clusters:
        print(f"{sum[i]} vulnerable docs assigned to cluster {i}.")
    else:
        print(f"{sum[i]} non-vulnerable docs assigned to cluster {i - num_clusters}.")

# Print cluster centers
#print("\nCluster Centers:")
#for i, center in enumerate(cluster_centers):
#    print(f"Cluster {i+1} center:", center)

'''
plt.figure(figsize=(4, 3), dpi=160)
x=document_vectors_tensor
plt.scatter(x[:, 0], x[:, 1], c=cluster_ids_x, cmap='cool')
#plt.scatter(y[:, 0], y[:, 1], c=cluster_ids_y, cmap='cool', marker='X')
plt.scatter(
    cluster_centers[:, 0], cluster_centers[:, 1],
    c='white',
    alpha=0.6,
    edgecolors='black',
    linewidths=2
)
plt.axis([-1, 1, -1, 1])
plt.tight_layout()
plt.savefig("my_out_3.png")
'''

#plt.figure(figsize=(10, 8))
#plt.scatter(document_vectors_tensor[:, 0], document_vectors_tensor[:, 1], c=cluster_ids_x, s = 50, cmap='viridis')
#plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, alpha = 0.75, marker='X')
#plt.title('K-means Clustering Result')

#plt.figure(figsize=(4, 3), dpi=160)
#plt.scatter(document_vectors_tensor[:, 0], document_vectors_tensor[:, 1], c=cluster_ids_x, cmap='cool')
#plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='white', alpha=0.6, edgecolors='black', linewidths=2)
#plt.tight_layout()
#plt.savefig("my_out_reentrancy.png")

'''
# Perform PCA using PyTorch
def pca_torch(X, num_components):
    X_mean = torch.mean(X, dim=0)
    X_centered = X - X_mean
    U, S, V = torch.svd(X_centered)
    return torch.matmul(X_centered, V[:, :num_components])

# Reduce dimensions to 2D
document_vectors_2d = pca_torch(document_vectors_tensor, 2)

# Verify all data points are retained after PCA
if document_vectors_2d.shape[0] != document_vectors_tensor.shape[0]:
    raise ValueError("PCA did not retain all data points.")

# Recalculate cluster centers in the reduced 2D space using PyTorch
cluster_ids_2d, cluster_centers_2d = kmeans(
    X=document_vectors_2d, 
    num_clusters=num_clusters, 
    distance='euclidean', 
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    iter_limit=max_iterations,
)

# Convert tensor to numpy for plotting
document_vectors_2d_np = document_vectors_2d.numpy()
cluster_centers_2d_np = cluster_centers_2d.numpy()

# Plot the 2D clusters
plt.figure(figsize=(4, 3), dpi=160)
plt.scatter(document_vectors_2d_np[:, 0], document_vectors_2d_np[:, 1], c=cluster_ids_2d, cmap='cool', s=30, alpha=0.6)
plt.scatter(cluster_centers_2d_np[:, 0], cluster_centers_2d_np[:, 1], c='white', alpha=1.0, edgecolors='black', linewidths=2, s=100)
plt.tight_layout()
plt.savefig("my_out_pca.png")

num_data_points = document_vectors_tensor.shape[0]
print(f"Number of data points: {num_data_points}")

# Print the number of data points
num_data_points = document_vectors_tensor.shape[0]
print(f"Number of data points: {num_data_points}")

# Verify the number of data points after PCA and clustering
print(f"Number of data points after PCA: {document_vectors_2d_np.shape[0]}")
print(f"Number of cluster centers: {cluster_centers_2d_np.shape[0]}")

print(document_vectors_2d_np)
'''
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
document_vectors_2d = tsne.fit_transform(document_vectors_tensor)

plt.figure(figsize=(6, 5), dpi=160)
plt.scatter(document_vectors_2d[:, 0], document_vectors_2d[:, 1], c=cluster_ids_x, cmap='cool', s=20, alpha=0.8)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='white', alpha=0.9, edgecolors='black', linewidths=2, s=100)

plt.title("t-SNE Visualization of Clusters")
plt.tight_layout()
plt.savefig("unchecked_low_level_calls_tsne.png")
plt.show()