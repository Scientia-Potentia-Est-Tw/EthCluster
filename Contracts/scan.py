import os
import subprocess
import re

contract_folder = '../Contracts/smartbugs-wild/contracts'
output_folder = '../Contracts/scan/smartbugs-wild'
os.makedirs(output_folder, exist_ok=True)

sol_files = [file for file in os.listdir(contract_folder) if file.endswith('.sol')]

def get_solidity_version(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    match = re.search(r'pragma solidity (\^|~|>=|<=|>|<)?(\d+\.\d+\.\d+);', content)
    return match.group(2) if match else None

def switch_solc_version(version):
    try:
        installed_versions_output = subprocess.check_output(['solc-select', 'versions'], text=True)
        installed_versions = [v.strip('* ').strip() for v in installed_versions_output.splitlines() if v.strip()]

        stable_version = version.replace("nightly", "")
        
        if '>=' in stable_version:
            base_version = stable_version.split('>=')[1]
            compatible_versions = [v for v in installed_versions if v >= base_version]
        elif '<=' in stable_version:
            base_version = stable_version.split('<=')[1]
            compatible_versions = [v for v in installed_versions if v <= base_version]
        elif '>' in stable_version:
            base_version = stable_version.split('>')[1]
            compatible_versions = [v for v in installed_versions if v > base_version]
        elif '<' in stable_version:
            base_version = stable_version.split('<')[1]
            compatible_versions = [v for v in installed_versions if v < base_version]
        elif '^' in stable_version:
            base_version = stable_version.split('^')[1]
            compatible_versions = [v for v in installed_versions if v.startswith(base_version.split('.')[0])]
        elif '~' in stable_version:
            base_version = stable_version.split('~')[1]
            compatible_versions = [v for v in installed_versions if v.startswith('.'.join(base_version.split('.')[:2]))]
        else:
            compatible_versions = [v for v in installed_versions if v == stable_version]

        if compatible_versions:
            selected_version = compatible_versions[0]
        else:
            selected_version = base_version
            subprocess.run(['solc-select', 'install', selected_version], check=True)

        if selected_version not in installed_versions:
            print(f'Solidity version {selected_version} is not installed. Installing now...')
            subprocess.run(['solc-select', 'install', selected_version], check=True)
        
        subprocess.run(['solc-select', 'use', selected_version, '--always-install'], check=True)
        print(f'Successfully switched to solc version: {selected_version}')
    
    except subprocess.CalledProcessError as e:
        print(f'Error switching solc version: {e}')
        print(f'Command output: {e.stdout}')
    except Exception as ex:
        print(f'An unexpected error occurred: {ex}')


def run_slither(file_path):
    contract_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(output_folder, f'{contract_name}.txt')
    try:
        result = subprocess.run(['slither', file_path, '--detect', 'reentrancy', '--detect', 'timestamp', 
                                 '--detect', 'tx-origin', '--detect', 'unchecked-lowlevel'], 
                                capture_output=True, text=True, check=False)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f'{result.stderr}')
        
        if os.path.exists(output_file):
            print(f'Output file created: {output_file}\n')
        else:
            print(f'Failed to create output file: {output_file}\n')

    except Exception as e:
        print(f'An unexpected error occurred while scanning {file_path}: {e}\n')

def main():
    for file_name in sol_files:
        file_path = os.path.join(contract_folder, file_name)

        if os.path.exists(file_path):
            version = get_solidity_version(file_path)
            if version:
                switch_solc_version(version)
                run_slither(file_path)
            else:
                print(f'No Solidity version found in {file_path}\n')
        else:
            print(f'File does not exist: {file_path}\n')

if __name__ == '__main__':
    main()
