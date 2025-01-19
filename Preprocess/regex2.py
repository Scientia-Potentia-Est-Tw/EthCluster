import os
import re

def detect_patterns_in_file(file_path):
    # Regex to detect trigger words
    trigger_pattern = re.compile(r'\b(call|send|transfer)\b', re.IGNORECASE)
    # Regex to detect balance keywords
    balance_pattern = re.compile(r'\b(balance|balances)\b', re.IGNORECASE)
    trigger_found = False  # State to remember if a trigger word was found

    try:
        with open(file_path, 'r') as file:
            for line in file:
                if trigger_found:
                    # If trigger word was found, look for balance keywords in subsequent lines
                    if balance_pattern.search(line):
                        return "detected"
                # Check if current line has a trigger word
                if trigger_pattern.search(line):
                    trigger_found = True

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")

    return "not detected"

def scan_files_up_to_n(n, directory_path):
    for i in range(1, n + 1):
        file_name = f"{i}.txt"
        file_path = os.path.join(directory_path, file_name)
        result = detect_patterns_in_file(file_path)
        print(f"File {file_name}: {result}")

# Example usage
#directory_path = '/home/antiransom/Unsupervised/Contracts/vuln/reentrancy/'  # Set the directory path where the files are located
directory_path = '/home/antiransom/Unsupervised/Contracts/clean/189/'
n = 189  # Set the number of files to read from 1 to n
scan_files_up_to_n(n, directory_path)
