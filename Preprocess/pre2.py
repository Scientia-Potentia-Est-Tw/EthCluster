import string
import re

def clean_and_save_sol_file(input_file_path, output_file_path):
    # Define a translation table to replace punctuation with space
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    try:
        with open(input_file_path, 'r') as file:
            content = file.read()
            # Remove single line and multi-line comments
            content = re.sub(r'//.*?\n|/\*.*?\*/', '', content, flags=re.DOTALL)
            # Replace punctuation with space
            no_punctuation = content.translate(translator)
            # Remove leading and trailing whitespace and replace newlines with spaces
            stripped_content = no_punctuation.strip().replace('\n', ' ')
            # Remove duplicate spaces
            single_spaced = re.sub(' +', ' ', stripped_content)

        # Split the content into words
        words = single_spaced.split()

        with open(output_file_path, 'w') as file:
            # Write each word on a new line
            for word in words:
                file.write(word + '\n')
        print(f"Processed file saved as {output_file_path}")
    except IOError as e:
        print(f"An error occurred: {e}")

# User input for file paths
input_file_path = input("Enter the path of your Solidity (.sol) file: ")
output_file_path = input("Enter the path for the output (.txt) file: ")

clean_and_save_sol_file(input_file_path, output_file_path)
