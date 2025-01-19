import string
import re
import os
from typing import Dict, Optional, Tuple
from gensim.models import Word2Vec, TfidfModel
from gensim.corpora import Dictionary
import numpy as np
import pickle

total_clean = 42
total_vuln = 18
total_file = total_clean + total_vuln
tfscore = 0.3
vsize = 50

########################################################################################################################################

# Assuming documents_path is the directory containing your text files
documents_path = '../Contracts/vuln/access_control'  # Replace with your actual directory path
documents_path2 = '../Contracts/clean/42'
documents = []

file_numbers_1 = []
for filename in os.listdir(documents_path):
    match = re.match(r'(\d+)\.txt', filename)
    if match:
        file_numbers_1.append((int(match.group(1)), filename))
        
file_numbers_2 = []
for filename in os.listdir(documents_path2):
    match = re.match(r'(\d+)\.txt', filename)
    if match:
        file_numbers_2.append((int(match.group(1)), filename))
        
file_numbers_1.sort()
file_numbers_2.sort()

for number, filename in file_numbers_1:
    #print(f"Reading file: {filename}")  # Optional: Print the filename being read
    with open(os.path.join(documents_path, filename), 'r') as file:
        document = file.read().splitlines()
        if not document:  # Check if the document is empty
            print(f"Empty file: {filename}")
            continue  # Skip this file
        documents.append(document)
        
for number, filename in file_numbers_2:
    #print(f"Reading file: {filename}")  # Optional: Print the filename being read
    with open(os.path.join(documents_path2, filename), 'r') as file:
        document = file.read().splitlines()
        if not document:  # Check if the document is empty
            print(f"Empty file: {filename}")
            continue  # Skip this file
        documents.append(document)
        
# Train the Word2Vec model
word2vec_model = Word2Vec(documents, vector_size=vsize, window=5,
                          min_count=1, workers=1, sg=1, epochs=10, seed=1337)

# Create a Gensim dictionary and corpus
dictionary = Dictionary(documents)
corpus = [dictionary.doc2bow(text) for text in documents]

# Train the TF-IDF model
tfidf_model = TfidfModel(corpus)

# Create a dictionary to hold word vectors weighted by TF-IDF
DocumentID = int
Word = str
TFIDFValue = float

class DocumentWordTFIDF:
    def __init__(self):
        self.data: Dict[DocumentID, Dict[Word, TFIDFValue]] = {}

    def add_document(self, doc_id: DocumentID):
        if doc_id not in self.data:
            self.data[doc_id] = {}

    def add_word(self, doc_id: DocumentID, word: Word, tfidf: TFIDFValue):
        if doc_id not in self.data:
            self.add_document(doc_id)
        self.data[doc_id][word] = tfidf

    def get_document_words(self, doc_id: DocumentID) -> Dict[Word, TFIDFValue]:
        return self.data.get(doc_id, {})

    def get_word_tfidf(self, doc_id: DocumentID, word: Word) -> TFIDFValue:
        return self.data.get(doc_id, {}).get(word, 0.0)

    def get_max_words(self, doc_id: DocumentID) -> Optional[Tuple[Word, TFIDFValue]]:
        words = self.get_document_words(doc_id)
        if not words:
            return None
        return max(words.items(), key=lambda x: x[1],)

    def __str__(self):
        return str(self.data)

word_vector_tfidf = {}
selected_words = {}
for i, doc in enumerate(corpus):
    doc_word_tfidf = DocumentWordTFIDF()
    selected_word = None
    for word_id, tfidf_score in tfidf_model[doc]:
        word = dictionary[word_id]
        if word not in word2vec_model.wv:
            continue
        if tfidf_score > tfscore:  # Filter words based on TF-IDF score
            word_vector_tfidf[word] = word2vec_model.wv[word]
            doc_word_tfidf.add_word(i + 1, word, tfidf_score)
        
    if selected_word is None:
        selected_word = doc_word_tfidf.get_max_words(i + 1)
    if selected_word:
        selected_words[i] = selected_word

# Creating Document Vectors
document_vectors = []
for i, doc in enumerate(documents):
    # Use a set to avoid duplicate words
    unique_words = set(doc)
    filtered_words = [ word for word in unique_words if word in word_vector_tfidf]
    print(f"doc {i+1}: {filtered_words}")
    selected_word = None

    if selected_word is None and filtered_words:
        selected_word = max(filtered_words, key=lambda w: word_vector_tfidf[w].max())
    
    if selected_word:
        document_vectors.append(word_vector_tfidf[selected_word])
        print(f"Document {i+1} selected word: {selected_word}")
    else:
        # Handle the case where no words meet the criteria
        zero_vector = np.zeros(word2vec_model.vector_size)
        document_vectors.append(zero_vector)
        print(f"No suitable words found for Document {i+1}, using zero vector.")

# Print the length of the final word_vector_tfidf dictionary and a sample TF-IDF score
print("\nLength of word_vector_tfidf:", len(word_vector_tfidf))

# Collect words and their highest TF-IDF scores
tfidf_scores = {}
counter = 0
for doc in corpus:
    for word_id, score in tfidf_model[doc]:
        word = dictionary[word_id]
        if score > tfscore and (word not in tfidf_scores or score > tfidf_scores[word]):
            tfidf_scores[word] = score
    counter += 1

# Sort the words by their TF-IDF scores in descending order
sorted_tfidf_scores = sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)

# Print the words and their scores
print("\nWords with TF-IDF score > " + str(tfscore) + " (sorted by score):")
for word, score in sorted_tfidf_scores:
    print(f"Word: {word}, Score: {score}")

#with open('../Contracts/access_control/document_vectors.pkl', 'wb') as f:
#    pickle.dump(document_vectors, f)
    
#word2vec_model.save("../Contracts/access_control/word2vec_model.model")
    
with open("../Contracts/vuln/access_control/mix/document_vectors.pkl", "wb") as f:
    pickle.dump(document_vectors, f)
    
word2vec_model.save("../Contracts/vuln/access_control/mix/word2vec_model.model")