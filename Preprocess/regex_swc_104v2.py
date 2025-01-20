import os
import re

prefixes = r'^\(.*(require|if|bool|success).*\)$'
postfixes = r'^.*\.(call|value|callcode|delegatecall|staticcall|send)\($'
vulnerable_pattern = r'^(?!.*\b(require|if|bool|success).*\.(call(?:\.value)?|callcode|delegatecall|staticcall|send)\().*\.(call(?:\.value)?|callcode|delegatecall|staticcall|send)\('
pattern = re.compile(vulnerable_pattern)
def detect_swc_104(contract_code):
    
    
    vulnerable_patterns = re.compile(rf'^{prefixes} (?:{postfixes})$')
    
    lines = contract_code.split('\n')
    for i, line in enumerate(lines):
        if re.search(pattern, line):
                return True, f"Possible SWC-104 Vulnerability in {i+1} row: {line.strip()}"
    return False, "Didn't check the SWC-104 vulnerability"

def scan_directory(directory):
    results = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.sol'):  # Assume all Solidity files ending .sol
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    contract_code = f.read()
                is_vulnerable, message = detect_swc_104(contract_code)
                results.append({
                    'file': file_path,
                    'is_vulnerable': is_vulnerable,
                    'message': message
                })
    return results

def main():
    directory = "../Contracts/vuln/unchecked_low_level_calls"
    #clean dataset
    #directory = "/home/antiransom/Unsupervised/Contracts/clean/122"
    results = scan_directory(directory)
    
    vulnerable_count = sum(1 for r in results if r['is_vulnerable'])
    
    for result in results:
        print(f"File: {result['file']}")
        print(f"Result: {'Vulnerable' if result['is_vulnerable'] else 'Safe'}")
        print(f"Detailed: {result['message']}")
        print("-" * 50)
    
    print(f"\nScan Done. Check amounts of {len(results)} files, find {vulnerable_count} potential SWC-104 vulnerability.\n")

if __name__ == "__main__":
    main()
