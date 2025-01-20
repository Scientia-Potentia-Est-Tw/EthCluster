import os
import re

def detect_swc_133(contract_code):
    
    vulnerable_patterns = [
        r"encodePacked\s*\(\s*[^,]+\s*,\s*[^)]+\s*\)"
    ]
    
    lines = contract_code.split('\n')
    for i, line in enumerate(lines):
        for pattern in vulnerable_patterns:
            if re.search(pattern, line):
                return True, f"Possible SWC-133 Vulnerability in {i+1} row: {line.strip()}"
    return False, "Didn't check the SWC-133 Vulnerability"

def scan_directory(directory):
    results = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.sol'):  # Assume all Solidity files use .sol ending
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    contract_code = f.read()
                is_vulnerable, message = detect_swc_133(contract_code)
                results.append({
                    'file': file_path,
                    'is_vulnerable': is_vulnerable,
                    'message': message
                })
    return results

def main():
    directory = "../Contracts/clean/262/"
    #clean dataset
    #directory = "/home/antiransom/Unsupervised/Contracts/clean"
    results = scan_directory(directory)
    
    vulnerable_count = sum(1 for r in results if r['is_vulnerable'])
    
    for result in results:
        print(f"File: {result['file']}")
        print(f"Result: {'Vulnerable' if result['is_vulnerable'] else 'Safe'}")
        print(f"Detailed: {result['message']}")
        print("-" * 50)
    
    print(f"\n Scan Done. Check amounts of {len(results)} files, find {vulnerable_count} potential SWC-133 vulnerability\n")

if __name__ == "__main__":
    main()
