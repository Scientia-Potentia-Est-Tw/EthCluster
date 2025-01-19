import string
import re
import os
from typing import Dict, Optional, Tuple
from gensim.models import Word2Vec, TfidfModel
from gensim.corpora import Dictionary
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


total_vuln = 52
total_clean = 122
total_file = total_clean + total_vuln
Vuln_regex_result = [0] * total_vuln
clean_regex_result = [0] * total_clean
tfscore = 0.7
vsize = 300

clean_path = "../Contracts/clean"
vuln_path = "../Contracts/vuln"
vulnerability = "unchecked_low_level_calls"

key_words = ["call", "callcode", "delegatecall", "staticcall", "send"]
prefixes = r'\b(require|if|bool|success)\b'
postfixes = r'\.(call|value|callcode|delegatecall|staticcall|send)\('
regex_pre = re.compile(prefixes)
regex_post = re.compile(postfixes)
########################################################################################################################################

def detect_swc_104(content):
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if regex_post.search(line):
            if not regex_pre.search(line):
                return True
    return False

def check_vulnerable_files():

    for i in range(1, total_vuln + 1):
        file_name = os.path.join(vuln_path, vulnerability, f"{i}.sol")
        if os.path.exists(file_name):
            with open(file_name, 'r', encoding='utf-8') as file:
                content = file.read()
                if detect_swc_104(content):
                    print(file_name+" detected\n")
                    vuln_regex_result[i - 1] = 1
                else:
                    print(file_name+" not detected\n")
                    vuln_regex_result[i - 1] = 0

def check_clean_files():
    for i in range(1, total_clean + 1):
        file_name = os.path.join(clean_path, str(total_clean), f"{i}.sol")
        if os.path.exists(file_name):
            with open(file_name, 'r', encoding='utf-8') as file:
                content = file.read()
                if detect_swc_104(content):
                    print(file_name+" detected\n")
                    clean_regex_result[i - 1] = 1
                else:
                    print(file_name+" not detected\n")
                    clean_regex_result[i - 1] = 0

vuln_regex_result = [0] * total_vuln
clean_regex_result = [0] * total_clean

check_vulnerable_files()
check_clean_files()

regex_result = [item for sublist in [vuln_regex_result, clean_regex_result] for item in sublist]

########################################################################################################################################

def clean_and_save_sol_file(input_file_path, output_file_path):
    # List of Solidity language keywords to remove
    solidity_keywords = [
        'pragma', 'import', 'contract', 'interface', 'library', 'struct', 
        'enum', 'function', 'event', 'error', 'using', 'for', 'constructor',
        'mapping', 'address', 'bool', 'string', 'var', 'bytes', 'uint', 'int', 
        'if', 'else', 'while', 'do', 'break', 'continue', 'return', 'throw', 
        'emit', 'public', 'private', 'internal', 'external', 'constant', 'immutable', 
        'view', 'pure', 'virtual', 'override', 'storage', 'memory', 'calldata', 
        'try', 'catch', 'revert', 'assert', 'require', 'new', 'delete', 'this', 'solidity'
    ]

    # Define a translation table to replace punctuation with space (excluding '/')
    translator = str.maketrans(string.punctuation.replace('/', ''), ' ' * (len(string.punctuation) - 1))

    # Ask user whether to remove comments
    remove_comments = True

    try:
        with open(input_file_path, 'r') as file:
            content = file.read()

        punctuation_present = True
        while punctuation_present:
            if remove_comments:
                # Remove single line and multi-line comments
                content = re.sub(r'//.*?\n|/\*.*?\*/', '', content, flags=re.DOTALL)
            else:
                # Replace comment symbols with space
                content = content.replace('//', ' ').replace('/*', ' ').replace('*/', ' ')
            # Replace punctuation with space
            no_punctuation = content.translate(translator)
            # Replace '/' with space
            no_slash = no_punctuation.replace('/', ' ')
            # Remove leading and trailing whitespace and replace newlines with spaces
            stripped_content = no_slash.strip().replace('\n', ' ')
            # Remove duplicate spaces
            single_spaced = re.sub(' +', ' ', stripped_content)
            # Split the content into words and remove Solidity keywords
            words = [word for word in single_spaced.split() 
                if word not in solidity_keywords and len(word) > 1 and not word.isdigit()]

            # Check for any remaining punctuation marks
            punctuation_present = any(char in string.punctuation for char in ' '.join(words))
            if punctuation_present:
                content = ' '.join(words)  # Prepare content for another cleaning iteration

        with open(output_file_path, 'w') as file:
            # Write each word on a new line
            for word in words:
                file.write(word + '\n')

    except IOError as e:
        print(f"An error occurred: {e}")
        
confirm_pre = True

if confirm_pre:
    # Loop to process files from 1 to X
    for i in range(1, total_vuln + 1):
        input_vuln_file_path = os.path.join(vuln_path, vulnerability, f"{i}.sol")
        output_vuln_file_path = os.path.join(vuln_path, vulnerability, f"{i}.txt")

        clean_and_save_sol_file(input_vuln_file_path, output_vuln_file_path)

    # Loop to process files from 1 to X
    for i in range(1, total_clean + 1):
        input_clean_file_path = os.path.join(clean_path, str(total_clean), f"{i}.sol")
        output_clean_file_path = os.path.join(clean_path, str(total_clean), f"{i}.txt")

        clean_and_save_sol_file(input_clean_file_path, output_clean_file_path)

########################################################################################################################################

# Assuming documents_path is the directory containing your text files
documents_path = "../Contracts/vuln/"+ vulnerability +"/"  # Replace with your actual directory path
documents_path2 = "../Contracts/clean/"+ str(total_clean) +"/"

# Read all documents
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
    with open(os.path.join(documents_path, filename), 'r') as file:
        document = file.read().splitlines()
        if not document:  # Check if the document is empty
            print(f"Empty file: {filename}")
            continue  # Skip this file
        documents.append(document)
        
for number, filename in file_numbers_2:
    with open(os.path.join(documents_path2, filename), 'r') as file:
        document = file.read().splitlines()
        if not document:  # Check if the document is empt
            continue  # Skip this file
        documents.append(document)

# Train the Word2Vec model
word2vec_model = Word2Vec(documents, vector_size=vsize, window=5,
                          min_count=1, workers=4, sg=1, epochs=10, seed=666)

# Create a Gensim dictionary and corpus
dictionary = Dictionary(documents)
corpus = [dictionary.doc2bow(text) for text in documents]

# Train the TF-IDF model
tfidf_model = TfidfModel(corpus)

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

# Create a dictionary to hold word vectors weighted by TF-IDF
word_vector_tfidf = {}
selected_words = {}
for i, doc in enumerate(corpus):
    doc_word_tfidf = DocumentWordTFIDF()
    selected_word = None
    for word_id, tfidf_score in tfidf_model[doc]:
        word = dictionary[word_id]
        if word not in word2vec_model.wv:
            continue
        if regex_result[i] == 1 and word in key_words:
            word_vector_tfidf[word] = word2vec_model.wv[word]
            selected_word = word
        if tfidf_score > tfscore:  # Filter words based on TF-IDF score
            word_vector_tfidf[word] = word2vec_model.wv[word]
            doc_word_tfidf.add_word(i + 1, word, tfidf_score)
        
    if selected_word is None:
        selected_word = doc_word_tfidf.get_max_words(i + 1)
    if selected_word:
        selected_words[i] = selected_word

#print(word_vector_tfidf)
# Creating Document Vectors
document_vectors = []
for i, doc in enumerate(documents):
    # Use a set to avoid duplicate words
    unique_words = set(doc)
    filtered_words = [ word for word in unique_words if word in word_vector_tfidf]
    print(f"doc {i+1}: {filtered_words}")
    selected_word = None
    
    for word in filtered_words:
        if word in key_words:
            selected_word = word
            break  

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
for i , doc in enumerate(corpus):
    for word_id, score in tfidf_model[doc]:
        word = dictionary[word_id]
        if score > tfscore and (word not in tfidf_scores or score > tfidf_scores[word]):
            tfidf_scores[word] = score
        if regex_result[i] == 1 and word in key_words:
            tfidf_scores[word] = score

# Sort the words by their TF-IDF scores in descending order
sorted_tfidf_scores = sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)

# Print the words and their scores
print("\nWords with TF-IDF score > " + str(tfscore) + " (sorted by score):")
for word, score in sorted_tfidf_scores:
    print(f"Word: {word}, Score: {score}")
    
with open("../Contracts/vuln/"+ vulnerability +"/mix/document_vectors.pkl", "wb") as f:
    pickle.dump(document_vectors, f)
    
word2vec_model.save("../Contracts/vuln/"+ vulnerability +"/mix/word2vec_model.model")

