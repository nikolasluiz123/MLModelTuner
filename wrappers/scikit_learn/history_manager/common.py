import json
import os
import pickle
from abc import abstractmethod, ABC
from typing import Generic

from sklearn.preprocessing import StandardScaler

from wrappers.scikit_learn.validator.common_validator import Result


class HistoryManager(ABC, Generic[Result]):
    """
    Classe utilizada para gerenciamento do histórico das execuções.

    Os resultados das execuções são salvos em um diretório especificado e em um arquivo JSON com o nome desejado. A
    estrutura do JSON é uma lista e os campos inseridos nele dependem do objeto `Result` e de quais campos desse objeto
    for julgado relevante manter no histórico.

    Além de salvar os dados do objeto de resultado da validação, também salvamos o modelo em si, utilizando o pickle. Dessa
    forma, é possível reutilizar o modelo treinado e validado para algum fim específico.
    """

    def __init__(self, output_directory: str, models_directory: str, params_file_name: str, cv_results_file_name: str):
        """
        Inicializa o HistoryManager com os diretórios e nome de arquivo apropriados.

        :param output_directory: Diretório de histórico que vai armazenar o arquivo JSON e os modelos.
        :param models_directory: Diretório específico para os modelos.
        :param params_file_name: Nome do arquivo JSON o qual conterá os parâmetros e resultados.
        :param cv_results_file_name: Nome do arquivo JSON que conterá as combinações de valores dos parâmetros da execução.
        """
        self.output_directory = output_directory
        self.models_directory = os.path.join(self.output_directory, models_directory)
        self.params_file_name = params_file_name
        self.cv_results_file_name = cv_results_file_name
        self._create_output_dir()

    @abstractmethod
    def save_result(self,
                    classifier_result: Result,
                    cv_results,
                    feature_selection_time: str,
                    search_time: str,
                    validation_time: str,
                    scoring: str,
                    features: list[str],
                    scaler: StandardScaler | None):
        """
        Função que deve ser implementada para salvar os dados do objeto `Result` no arquivo JSON.

        :param classifier_result: Objeto com os dados da validação do melhor modelo encontrado.
        :param cv_results: Dicionário obtido da implementação de busca de parâmetros com as combinações testadas.
        :param feature_selection_time: Implementação de seleção de features utilizada.
        :param search_time: Tempo que demorou o processamento de busca de parâmetros.
        :param validation_time: Tempo que demorou o processamento de validação do melhor modelo.
        :param scoring: Métrica de validação utilizada.
        :param features: Features selecionadas pela implementação.
        :param scaler: Implementação opcional utilizada para escalar os dados
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

    def has_history(self) -> bool:
        """
        Verifica se já existem entradas no histórico de resultados.

        :return: True se o histórico contiver entradas, caso contrário False.
        """
        output_path = os.path.join(self.output_directory, f"{self.params_file_name}.json")

        if not os.path.exists(output_path):
            return False

        with open(output_path, 'r') as file:
            data = json.load(file)
            return len(data) > 0

    @abstractmethod
    def load_validation_result_from_history(self, index: int = -1) -> Result:
        """
        Função abstrata para carregar um resultado de validação do histórico.

        :param index: Índice do resultado a ser carregado. Se -1, o último resultado é retornado.
        :return: O objeto Result correspondente ao histórico solicitado.
        """

    def get_dictionary_from_params_json(self, index: int):
        return self._get_dictionary_from_json(index, self.params_file_name)

    def get_dictionary_from_cv_results_json(self, index: int):
        return self._get_dictionary_from_json(index, self.cv_results_file_name)

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
            raise FileNotFoundError(
                f"O arquivo {file_name}.json não foi encontrado no diretório {self.output_directory}.")

        with open(output_path, 'r') as file:
            data = json.load(file)

        if index < -1 or index >= len(data):
            raise IndexError(f"Índice {index} fora dos limites. O arquivo contém {len(data)} entradas.")

        result_dict = data[index]

        return result_dict

    def _save_model(self, estimator):
        """
        Salva o modelo treinado utilizando pickle.

        O modelo é salvo em um arquivo .pkl no diretório específico para modelos,
        com um nome baseado no tamanho do histórico atual.

        :param estimator: O modelo a ser salvo.
        """
        history_len = self._get_history_len()
        output_path = os.path.join(self.models_directory, f"model_{history_len}.pkl")

        with open(output_path, 'wb') as file:
            pickle.dump(estimator, file)

    def get_saved_model(self, version: int):
        """
        Recupera um modelo salvo a partir de seu índice/version.

        :param version: O índice do modelo a ser recuperado.
        :return: O modelo recuperado.
        :raises FileNotFoundError: Se o modelo não for encontrado.
        """
        output_path = os.path.join(self.models_directory, f"model_{version}.pkl")

        if not os.path.exists(output_path):
            raise FileNotFoundError(
                f"O modelo versão {version} não foi encontrado no diretório {self.models_directory}.")

        with open(output_path, 'rb') as f:
            return pickle.load(f)

    def _get_history_len(self) -> int:
        """
        Retorna o número de entradas no histórico de resultados.

        :return: O comprimento do histórico (número de entradas no arquivo JSON).
        """
        output_path = os.path.join(self.output_directory, f"{self.params_file_name}.json")

        if not os.path.exists(output_path):
            return 0

        with open(output_path, 'r') as file:
            data = json.load(file)
            return len(data)
