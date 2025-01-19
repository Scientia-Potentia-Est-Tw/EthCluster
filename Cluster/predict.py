import os
import re
from matplotlib import pyplot as plt
import torch
from gensim.models import Word2Vec, TfidfModel
from gensim.corpora import Dictionary
import pickle
import numpy as np
from kmeans_pytorch import kmeans

# 請依據訓練模型修改參數
vulnerable = "timestamp"
running_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Word2Vec.load(f"../Contracts/vuln/{vulnerable}/mix/word2vec_model.model")
data_path = "../Contracts/predict/test"
cluster_centers = torch.load(f"../Contracts/vuln/{vulnerable}/mix/cluster_centers").to(running_device)
num_clusters = 5
cluster_elements = [[] for _ in range(num_clusters)]
count_elements = [0] * num_clusters

def extract_number(filename):
    # 從檔名中提取數字
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

def get_average_vector(words):
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    
    # 計算詞向量的平均值來表示文本
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

def predict_cluster(data_vector, cluster_centers):
    vector_tensor = torch.tensor(data_vector, dtype=torch.float32, device=running_device)
    distances = torch.norm(cluster_centers - vector_tensor, dim=1)
    return torch.argmin(distances).item()

def add_element(file, cluster_number):
    cluster_elements[cluster_number].append(file)
    count_elements[cluster_number] += 1

def print_cluster_elements(clusters):
    for index, item in enumerate(clusters):
        print(f"{count_elements[index]} elements in cluster {index}:\n{item}\n")

def print_elements_count(clusters):
    for index, _ in enumerate(clusters):
        print(f"{count_elements[index]} elements in cluster {index}")

def main():
    for filename in sorted(os.listdir(data_path), key=extract_number):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(data_path, filename), 'r') as file:
            vector = file.read().splitlines()
            cluster = predict_cluster(get_average_vector(vector), cluster_centers) 
            add_element(filename, cluster)

    print_cluster_elements(cluster_elements)
    print_elements_count(cluster_elements)

if __name__ == "__main__":
    main()
