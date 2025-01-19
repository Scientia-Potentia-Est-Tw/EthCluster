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

# 假設 n 是你要讀取的檔案數量
n = 523  # 例如你有 10 個 .sol 檔案
check_files(n)
