from gensim.models import Word2Vec

# Assuming output_file_path is defined and points to your processed text file
output_file_path = 'test2.txt'  # Replace with your actual file path

# Read the processed file
with open(output_file_path, 'r') as file:
    words = file.read().splitlines()

# Prepare the data for Word2Vec (a list of lists of words)
data_for_word2vec = [words]  # In this case, it's a single 'sentence'

# User choice for model training: Skip-gram (1) or CBOW (0)
# model_choice = int(input("Enter 1 for Skip-gram model or 0 for CBOW model: "))

# Train the Word2Vec model based on user choice
model = Word2Vec(data_for_word2vec, vector_size=300, window=5, min_count=1, workers=4, sg=1, epochs=10)

# Displaying vectors for words in the model's vocabulary
print("\nVectors for words in the model's vocabulary:")
for word in model.wv.index_to_key:
    vector = model.wv[word]
    print(f"Word: {word}\nVector: {vector}\n")

vocabulary = list(model.wv.index_to_key)
print(f"\nVocabulary: {vocabulary[:2000]}")
print(f"Total number of words in the model's vocabulary: {len(model.wv.index_to_key)}")