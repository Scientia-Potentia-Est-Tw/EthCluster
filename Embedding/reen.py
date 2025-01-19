import os
import re

def detect_patterns_in_file(file_path):
    # Regex to detect trigger words
    trigger_pattern = re.compile(r'\b(call)\b', re.IGNORECASE)
    # Regex to detect balance keywords
    balance_pattern = re.compile(r'\b(balance|balances)\b', re.IGNORECASE)
    line_buffer = []  # Buffer to keep track of the next 5 lines after a trigger word is found

    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Check if there are lines in the buffer to examine
                if line_buffer:
                    line_buffer.append(line)
                    # Check all lines in the buffer for balance keywords
                    if any(balance_pattern.search(l) for l in line_buffer):
                        return "detected"
                    # If buffer exceeds 5 lines, drop the oldest (shift the window)
                    if len(line_buffer) > 5:
                        line_buffer.pop(0)
                # Check if current line has a trigger word
                if trigger_pattern.search(line):
                    line_buffer = [line]  # Reset buffer with current line

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
#n = 81  # Set the number of files to read from 1 to n
n = 189
scan_files_up_to_n(n, directory_path)
