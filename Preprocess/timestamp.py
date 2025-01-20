import re
import os

def check_files(n):
    pattern = re.compile(r"((\bnow\b)|(\bblock\.timestamp\b))")

    for i in range(1, n + 1):
        file_name = f"/home/antiransom/Unsupervised/Contracts/clean/{i}.sol"
        if os.path.exists(file_name):
            with open(file_name, 'r', encoding='utf-8') as file:
                content = file.read()
                if pattern.search(content):
                    print(f"File {file_name}: Detected.")
                else:
                    print(f"File {file_name}: Not detected.")

# Assume n is the number of files what you want read
n = 523  # Assume you would like read 10 .sol files
check_files(n)
