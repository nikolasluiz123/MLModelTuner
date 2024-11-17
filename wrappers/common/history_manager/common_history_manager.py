import json
import os
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from wrappers.common.validator.results.common_validation_result import CommonValidationResult

CommonValResult = TypeVar('CommonValResult', bound=CommonValidationResult)

class CommonHistoryManager(ABC, Generic[CommonValResult]):
    """
    Implementação base para o gerenciadores de histórico, contendo as implementações padrões.

    A manutenção do histórico é baseada em persistir os dados das execuções em uma lista no formato JSON,
    dessa forma, deve ser possível reproduzir uma execução que esteja nesse histórico com o intuito de visualizar novamente
    os dados. Além disso, se for preciso criar algum tipo de análise dos dados das execuções realizadas seria possível
    criar uma implementação específica para isso e os dados viriam desse histórico.

    Além de manter os dados das execuções mantemos os modelos treinados e validados pelos processos internos, dessa forma,
    é possível recuperar esse modelo e utilizá-lo para algum fim específico.
    """
    def __init__(self,
                 output_directory: str,
                 models_directory: str,
                 best_params_file_name: str):
        """
        :param output_directory: Diretório principal onde todos os dados do histórico serão armazenados
        :param models_directory: Diretório criado abaixo de `output_directory` especificamente para armazenar os modelos
        :param best_params_file_name: Nome do arquivo JSON que será criado para armazenar a lista de execuções
        """

        self.output_directory = output_directory
        self.models_directory = os.path.join(self.output_directory, models_directory)
        self.best_params_file_name = best_params_file_name

        self._create_output_dir()

    @abstractmethod
    def load_validation_result_from_history(self, index: int = -1) -> CommonValResult:
        """
        Função abstrata para carregar um resultado de validação do histórico. Deve retornar o objeto com os dados da
        validação do modelo que foi previamente executada a partir dos dados históricos.

        :param index: Índice do resultado a ser carregado. Por padrão retorna o último resultado da lista que
                      representaria a execução mais recente
        """

    @abstractmethod
    def _save_model(self, model):
        """
        Função que deve ser implementada com os processos necessários para salvar o modelo treinado e validado em um
        arquivo no formado que funciona melhor para cada biblioteca.

        :param model Representação do modelo da biblioteca que será salvo como arquivo
        """

    @abstractmethod
    def get_saved_model(self, version: int):
        """
        Função responsável por retornar um modelo que já tenha sido salvo como arquivo.

        :param version: Versão do modelo que deseja recuperar, isso normalmente é concatenado no nome do arquivo ao salvar.
        """

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

    def get_dictionary_from_params_json(self, index: int) -> dict:
        """
        Função que retorna um dicionário que contém os dados de uma execução específica realizada.
        """
        return self._get_dictionary_from_json(index, self.best_params_file_name)

    def _get_dictionary_from_json(self, index: int, file_name: str) -> dict:
        """
        Retorna um dicionário de um índice específico obtido de um arquivo JSON que representa uma lista.

        :param index: Índice do resultado a ser recuperado.
        :param file_name: Nome do arquivo JSON a ser utilizado.

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
        Retorna se já existem entradas no histórico de resultados.
        """
        return self.get_history_len() > 0

    def get_history_len(self) -> int:
        """
        Retorna o número de entradas no histórico de resultados.
        """
        output_path = os.path.join(self.output_directory, f"{self.best_params_file_name}.json")

        if not os.path.exists(output_path):
            return 0

        with open(output_path, 'r') as file:
            data = json.load(file)
            return len(data)