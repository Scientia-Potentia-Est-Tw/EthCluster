from gensim.models import Word2Vec, TfidfModel
from gensim.corpora import Dictionary
import numpy as np

# Assuming output_file_path is defined and points to your processed text file
output_file_path = 'test2.txt'  # Replace with your actual file path

# Read the processed file
with open(output_file_path, 'r') as file:
    words = file.read().splitlines()

# Prepare the data for Word2Vec (a list of lists of words)
data_for_word2vec = [words]  # In this case, it's a single 'sentence'

# Train the Word2Vec model
# model_choice = int(input("Enter 1 for Skip-gram model or 0 for CBOW model: "))
word2vec_model = Word2Vec(data_for_word2vec, vector_size=100, window=5, min_count=1, workers=4, sg=1, epochs=10)

data_for_word2vec = [line.split() for line in words]

# Create a Gensim dictionary and corpus
dictionary = Dictionary(data_for_word2vec)
print("Dictionary:", dictionary.token2id)
corpus = [dictionary.doc2bow(text) for text in data_for_word2vec]
print("Corpus:", corpus)

# Train the TF-IDF model
tfidf_model = TfidfModel(corpus)

# Debugging: Print TF-IDF scores for the first document
print("\nTF-IDF scores for the first document:")
for word_id, score in tfidf_model[corpus[0]]:
    print(f"Word: {dictionary[word_id]}, Score: {score}")

# Create a dictionary to hold word vectors weighted by TF-IDF
word_vector_tfidf = {}
for word_id, word in dictionary.iteritems():
    if word in word2vec_model.wv:
        tfidf_scores = dict(tfidf_model[corpus[0]])
        tfidf_score = tfidf_scores.get(word_id, 0)
        word_vector_tfidf[word] = word2vec_model.wv[word] * tfidf_score

# Displaying vectors for words in the model's vocabulary, weighted by TF-IDF
print("\nTF-IDF weighted vectors for words in the model's vocabulary:")
for word, vector in word_vector_tfidf.items():
    print(f"Word: {word}\nVector: {vector}\n")

# Print the total number of words in the model's vocabulary
print(f"Total number of words in the model's vocabulary: {len(word2vec_model.wv.index_to_key)}")
