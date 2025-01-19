import torch
import pickle
from kmeans_pytorch import kmeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the first set of document vectors
#with open('../Contracts/access_control/document_vectors.pkl', 'rb') as f:
#    document_vectors1 = pickle.load(f)

# Load the second set of document vectors
#with open('../Contracts/clean/document_vectors.pkl', 'rb') as f:
#    document_vectors2 = pickle.load(f)
    
with open('../Contracts/vuln/access_control/mix_new/document_vectors.pkl', 'rb') as f:
    document_vectors = pickle.load(f)

# Combine the two sets of vectors if needed
#document_vectors = document_vectors1 + document_vectors2

# Convert document vectors to PyTorch tensor
document_vectors_tensor = torch.tensor(document_vectors, dtype=torch.float32)

# Number of clusters - adjust based on your needs
num_clusters = 5

# Set the maximum number of iterations
max_iterations = 1000

# Perform K-means clustering
cluster_ids_x, cluster_centers = kmeans(
    X=document_vectors_tensor, 
    num_clusters=num_clusters, 
    distance='euclidean', 
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    iter_limit=max_iterations
)

print(cluster_ids_x)

# Print cluster assignments
print("\nCluster assignments:")
for i, cluster_id in enumerate(cluster_ids_x):
    print(f"Document {i+1} assigned to cluster {cluster_id}")

# Print cluster centers
print("\nCluster Centers:")
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i+1} center:", center)

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
plt.axis([-0.3, 0.2, -0.2, 0.2])
plt.tight_layout()
plt.savefig("my_out.png")