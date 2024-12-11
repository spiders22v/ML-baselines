import ast
import os

def extract_requirements(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        tree = ast.parse(file.read(), filename=file_path)

    requirements = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                package_name = alias.name.split('.')[0]
                requirements.add(package_name)
        elif isinstance(node, ast.ImportFrom):
            package_name = node.module.split('.')[0]
            requirements.add(package_name)
                
    return requirements

def create_requirements_file(code_file_path, output_path='requirements.txt'):
    requirements = extract_requirements(code_file_path)

    with open(output_path, 'w', encoding='utf-8') as output_file:
        for requirement in sorted(requirements):
            output_file.write(f"{requirement}\n")

if __name__ == "__main__":
    # 입력할 코드 파일의 상대 경로를 지정하세요.
    code_file_path = 'dockerfileGenerator_qt5_v2.py'

    # 현재 디렉토리를 기준으로 상대 경로를 절대 경로로 변환
    absolute_code_file_path = os.path.abspath(code_file_path)

    create_requirements_file(absolute_code_file_path)   
    
    
 
