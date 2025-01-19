from gensim.models import Word2Vec, TfidfModel
from gensim.corpora import Dictionary
import numpy as np
import os

# Assuming documents_path is the directory containing your text files
documents_path = '../Contracts/access_control'  # Replace with your actual directory path

# Read all documents
documents = []
for filename in os.listdir(documents_path):
    if filename.endswith('.txt'):  # Assuming text files
        with open(os.path.join(documents_path, filename), 'r') as file:
            # print("Reading file:", filename)
            document = file.read().splitlines()
            documents.append(document)

# Flatten the list of documents for Word2Vec training
# data_for_word2vec = [word for doc in documents for word in doc]

# Train the Word2Vec model
# model_choice = int(input("Enter 1 for Skip-gram model or 0 for CBOW model: "))
# word2vec_model = Word2Vec(data_for_word2vec, vector_size=200, window=5, min_count=1, workers=4, sg=1, epochs=10)
word2vec_model = Word2Vec(documents, vector_size=200, window=5, min_count=1, workers=4, sg=1, epochs=10)

# Create a Gensim dictionary and corpus
dictionary = Dictionary(documents)
corpus = [dictionary.doc2bow(text) for text in documents]

# Train the TF-IDF model
tfidf_model = TfidfModel(corpus)

# Create a dictionary to hold word vectors weighted by TF-IDF
word_vector_tfidf = {}
for doc in corpus:
    for word_id, tfidf_score in tfidf_model[doc]:
        word = dictionary[word_id]
        if word in word2vec_model.wv:
            word_vector_tfidf[word] = word2vec_model.wv[word] * tfidf_score

# Displaying vectors for words in the model's vocabulary, weighted by TF-IDF
print("\nTF-IDF weighted vectors for words in the model's vocabulary:")
for word, vector in word_vector_tfidf.items():
    print(f"Word: {word}\nVector: {vector}\n")

print("Length of word_vector_tfidf:", len(word_vector_tfidf))
#print("Sample from word_vector_tfidf:", list(word_vector_tfidf.items())[:5])

print("\nTF-IDF scores for a sample document:")
sample_doc = corpus[4]  # Adjust index to view different documents
for word_id, score in tfidf_model[sample_doc]:
    print(f"Word: {dictionary[word_id]}, Score: {score}")
