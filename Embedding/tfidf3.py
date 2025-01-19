from gensim.models import Word2Vec, TfidfModel
from gensim.corpora import Dictionary
import numpy as np
import os
# import torch  # Import torch for later use

# Assuming documents_path is the directory containing your text files
documents_path = '../Contracts/access_control'  # Replace with your actual directory path

# Read all documents
documents = []
for filename in os.listdir(documents_path):
    if filename.endswith('.txt'):  # Assuming text files
        with open(os.path.join(documents_path, filename), 'r') as file:
            document = file.read().splitlines()
            documents.append(document)

# Train the Word2Vec model
word2vec_model = Word2Vec(documents, vector_size=100, window=5, min_count=1, workers=4, sg=1, epochs=10)

# Create a Gensim dictionary and corpus
dictionary = Dictionary(documents)
corpus = [dictionary.doc2bow(text) for text in documents]

# Train the TF-IDF model
tfidf_model = TfidfModel(corpus)

# Create a dictionary to hold word vectors weighted by TF-IDF
word_vector_tfidf = {}
for doc in corpus:
    for word_id, tfidf_score in tfidf_model[doc]:
        if tfidf_score > 0.4:  # Filter words based on TF-IDF score
            word = dictionary[word_id]
            if word in word2vec_model.wv:
                word_vector_tfidf[word] = word2vec_model.wv[word] * tfidf_score

# Creating Document Vectors
document_vectors = []
for i, doc in enumerate(documents):
    # Use a set to avoid duplicate words
    unique_words = set(doc)
    filtered_vectors = [(word, word_vector_tfidf[word]) for word in unique_words if word in word_vector_tfidf]

    # Print the filtered vectors for each document
    print(f"Document {i+1} filtered vectors:")
    for word, vec in filtered_vectors:
        print(f"Word: {word}, Vector: {vec}")

    if filtered_vectors:  # Check if filtered_vectors is not empty
        # Extract only the vectors for averaging
        vectors_only = [vec for _, vec in filtered_vectors]
        doc_vector = np.mean(vectors_only, axis=0)
        document_vectors.append(doc_vector)
    else:
        # Handle the case where no words meet the threshold, e.g., by appending a zero vector
        zero_vector = np.zeros(word2vec_model.vector_size)
        document_vectors.append(zero_vector)
        print(f"No words meeting TF-IDF threshold for Document {i+1}, using zero vector.")

# Convert document vectors to PyTorch tensor
# document_vectors_tensor = torch.tensor(document_vectors, dtype=torch.float32)

# Print the length of the final word_vector_tfidf dictionary and a sample TF-IDF score
print("\nLength of word_vector_tfidf:", len(word_vector_tfidf))

# Collect words and their highest TF-IDF scores
tfidf_scores = {}

for doc in corpus:
    for word_id, score in tfidf_model[doc]:
        word = dictionary[word_id]
        if score > 0.4 and (word not in tfidf_scores or score > tfidf_scores[word]):
            tfidf_scores[word] = score

# Sort the words by their TF-IDF scores in descending order
sorted_tfidf_scores = sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)

# Print the words and their scores
print("\nWords with TF-IDF score > 0.4 (sorted by score):")
for word, score in sorted_tfidf_scores:
    print(f"Word: {word}, Score: {score}")
