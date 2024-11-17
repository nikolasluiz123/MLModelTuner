import json
import os
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from wrappers.common.validator.results.common_validation_result import CommonValidationResult

CommonValResult = TypeVar('CommonValResult', bound=CommonValidationResult)

class CommonHistoryManager(ABC, Generic[CommonValResult]):

    def __init__(self,
                 output_directory: str,
                 models_directory: str,
                 best_params_file_name: str):
        self.output_directory = output_directory
        self.models_directory = os.path.join(self.output_directory, models_directory)
        self.best_params_file_name = best_params_file_name

        self._create_output_dir()

    @abstractmethod
    def load_validation_result_from_history(self, index: int = -1) -> CommonValResult:
        """
        Função abstrata para carregar um resultado de validação do histórico.

        :param index: Índice do resultado a ser carregado. Se -1, o último resultado é retornado.
        :return: O objeto Result correspondente ao histórico solicitado.
        """

    @abstractmethod
    def _save_model(self, model):
        ...

    @abstractmethod
    def get_saved_model(self, version: int):
        ...

    def _create_output_dir(self):
        """
        Cria os diretórios de saída para o histórico e modelos, caso não existam.
        """
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        if not os.path.exists(self.models_directory):
            os.makedirs(self.models_directory)

    def _save_dictionary_in_json(self, dictionary, file_name: str):
        """
        Salva um dicionário em formato JSON, adicionando os dados ao arquivo existente.

        :param dictionary: Dicionário contendo os dados a serem salvos no arquivo JSON.
        """
        output_path = os.path.join(self.output_directory, f"{file_name}.json")

        if os.path.exists(output_path):
            with open(output_path, 'r') as file:
                data = json.load(file)
        else:
            data = []

        data.append(dictionary)

        with open(output_path, 'w') as file:
            json.dump(data, file, indent=4)

    def get_dictionary_from_params_json(self, index: int):
        return self._get_dictionary_from_json(index, self.best_params_file_name)

    def _get_dictionary_from_json(self, index: int, file_name: str):
        """
        Obtém um dicionário de resultados a partir do arquivo JSON.

        :param index: Índice do resultado a ser recuperado.
        :param file_name: Nome do arquivo JSON a ser utilizado.
        :return: Dicionário contendo os dados do resultado.
        :raises FileNotFoundError: Se o arquivo JSON não for encontrado.
        :raises IndexError: Se o índice estiver fora dos limites do histórico.
        """
        output_path = os.path.join(self.output_directory, f"{file_name}.json")

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"O arquivo {file_name}.json não foi encontrado no diretório {self.output_directory}.")

        with open(output_path, 'r') as file:
            data = json.load(file)

        if index >= len(data):
            raise IndexError(f"Índice {index} fora dos limites. O arquivo contém {len(data)} entradas.")

        result_dict = data[index]

        return result_dict

    def has_history(self) -> bool:
        """
        Verifica se já existem entradas no histórico de resultados.

        :return: True se o histórico contiver entradas, caso contrário False.
        """
        return self.get_history_len() > 0

    def get_history_len(self) -> int:
        """
        Retorna o número de entradas no histórico de resultados.

        :return: O comprimento do histórico (número de entradas no arquivo JSON).
        """
        output_path = os.path.join(self.output_directory, f"{self.best_params_file_name}.json")

        if not os.path.exists(output_path):
            return 0

        with open(output_path, 'r') as file:
            data = json.load(file)
            return len(data)