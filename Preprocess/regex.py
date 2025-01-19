import os
import re

def detect_patterns_in_file(file_path):
    pattern = re.compile(r'\b(call|send|transfer)\b.*\b(balance|balances)\b', re.IGNORECASE)
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if pattern.search(line):
                    return "detected"
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
directory_path = '/home/antiransom/Unsupervised/Contracts/vuln/reentrancy/'  # Set the directory path where the files are located
n = 81  # Set the number of files to read from 1 to n
scan_files_up_to_n(n, directory_path)
