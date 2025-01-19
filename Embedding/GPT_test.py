import os
import re
import string
from gensim.models import Word2Vec, TfidfModel
from gensim.corpora import Dictionary
import numpy as np
from collections import defaultdict
import pickle

# Helper functions and preprocessing routines
def clean_text(content, remove_comments=True):
    translator = str.maketrans(string.punctuation.replace('/', ''), ' ' * (len(string.punctuation) - 1))
    solidity_keywords = set([
        'pragma', 'import', 'contract', 'interface', 'library', 'struct', 
        'enum', 'function', 'event', 'error', 'using', 'for', 'constructor',
        'mapping', 'address', 'bool', 'string', 'var', 'bytes', 'uint', 'int', 
        'if', 'else', 'while', 'do', 'break', 'continue', 'return', 'throw', 
        'emit', 'public', 'private', 'internal', 'external', 'constant', 'immutable', 
        'view', 'pure', 'virtual', 'override', 'storage', 'memory', 'calldata', 
        'try', 'catch', 'revert', 'assert', 'require', 'new', 'delete', 'this', 'solidity'  # truncated for brevity
    ])

    if remove_comments:
        content = re.sub(r'//.*?\n|/\*.*?\*/', '', content, flags=re.DOTALL)
    no_punctuation = content.translate(translator)
    no_slash = no_punctuation.replace('/', ' ')
    stripped_content = no_slash.strip().replace('\n', ' ')
    single_spaced = re.sub(' +', ' ', stripped_content)
    words = [word for word in single_spaced.split() if word not in solidity_keywords and len(word) > 1 and not word.isdigit()]
    return words

def detect_patterns_in_text(text, trigger_pattern, balance_pattern):
    for line in text:
        if trigger_pattern.search(line):
            next_lines = text[text.index(line):text.index(line) + 6]  # Get 5 lines forward
            if any(balance_pattern.search(l) for l in next_lines):
                return True
    return False

# Settings and initialization
directory_path = "/home/antiransom/Unsupervised/Contracts/vuln/reentrancy/"
n = 189
documents = []
trigger_pattern = re.compile(r'\b(call)\b', re.IGNORECASE)
balance_pattern = re.compile(r'\b(balance|balances)\b', re.IGNORECASE)

# Read, process and detect files
for i in range(1, n + 1):
    file_name = f"{i}.sol"
    file_path = os.path.join(directory_path, file_name)
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        processed_words = clean_text(content)
        pattern_detected = detect_patterns_in_text(processed_words, trigger_pattern, balance_pattern)
        if pattern_detected:
            processed_words.append('call')
        documents.append(processed_words)
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")

# Continue with Word2Vec and TF-IDF
word2vec_model = Word2Vec(documents, vector_size=100, window=5, min_count=1, workers=4, sg=1, epochs=10)
dictionary = Dictionary(documents)
corpus = [dictionary.doc2bow(text) for text in documents]
tfidf_model = TfidfModel(corpus)

# Calculate Document Vectors
document_vectors = []
for doc in documents:
    doc_bow = dictionary.doc2bow(doc)
    doc_tfidf = tfidf_model[doc_bow]
    doc_vec = np.zeros(word2vec_model.vector_size)
    for word_id, tfidf_val in doc_tfidf:
        word = dictionary[word_id]
        if tfidf_val > 0.5 and word in word2vec_model.wv:
            doc_vec += word2vec_model.wv[word] * tfidf_val
    doc_vec = doc_vec / len(doc_tfidf) if doc_tfidf else np.zeros(word2vec_model.vector_size)
    document_vectors.append(doc_vec)

# Optionally save the model and vectors
with open("/home/antiransom/Models/document_vectors.pkl", 'wb') as f:
    pickle.dump(document_vectors, f)
word2vec_model.save("/home/antiransom/Models/word2vec_model.model")

print("Processing complete.")