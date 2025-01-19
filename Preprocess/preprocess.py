import os
import string
import re

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
    translator = str.maketrans(string.punctuation.replace('/', ''), 
                                ' ' * (len(string.punctuation) - 1))

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

        # Write cleaned words to the output file
        with open(output_file_path, 'w') as file:
            for word in words:
                file.write(word + '\n')
                
    except IOError as e:
        print(f"An error occurred: {e}")

# Define the directory containing .sol files
directory_path = "../Contracts/slither_filtered"

# Loop through all .sol files in the specified directory
for filename in os.listdir(directory_path):
    if filename.endswith('.sol'):
        input_file_path = os.path.join(directory_path, filename)
        
        # Define output file path (same name but .txt extension)
        output_file_path = os.path.join(directory_path, f"{os.path.splitext(filename)[0]}.txt")
        
        clean_and_save_sol_file(input_file_path, output_file_path)