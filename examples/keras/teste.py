import os

directory = 'search_hp_history'
projects = os.listdir(directory)
project_names = [p for p in projects if os.path.isdir(os.path.join(directory, p))]
last_project = project_names[-1]

oracle_file_path = os.path.join(directory, last_project, 'oracle.json')

if os.path.isfile(oracle_file_path):
    with open(oracle_file_path, 'r') as file:
        oracle_data = file.read()
        print(oracle_data)  # Exibe o conteúdo do arquivo
else:
    print("O arquivo 'oracle.json' não foi encontrado.")