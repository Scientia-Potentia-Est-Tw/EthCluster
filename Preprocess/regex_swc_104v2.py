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
                return True, f"可能的SWC-104漏洞在第{i+1}行: {line.strip()}"
    return False, "沒有檢測到明顯的SWC-104漏洞"

def scan_directory(directory):
    results = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.sol'):  # 假設所有的Solidity文件都以.sol結尾
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
        print(f"文件: {result['file']}")
        print(f"結果: {'易受攻擊' if result['is_vulnerable'] else '未檢測到漏洞'}")
        print(f"詳情: {result['message']}")
        print("-" * 50)
    
    print(f"\n掃描完成。檢查了 {len(results)} 個文件，發現 {vulnerable_count} 個潛在的SWC-104漏洞。\n")

if __name__ == "__main__":
    main()