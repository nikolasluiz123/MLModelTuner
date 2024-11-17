import os
import pickle
from abc import abstractmethod, ABC

from sklearn.preprocessing import StandardScaler

from wrappers.common.history_manager.common_history_manager import CommonHistoryManager, CommonValResult


class ScikitLearnCommonHistoryManager(CommonHistoryManager[CommonValResult], ABC):
    """
    Implementação comum utilizada para manipular os dados históricos das execuções dos modelos baseados no scikit-learn.
    """

    def __init__(self, output_directory: str, models_directory: str, best_params_file_name: str, cv_results_file_name: str):
        """
        :param cv_results_file_name: Nome do arquivo JSON que conterá as combinações de valores dos parâmetros testados
                                     durante a busca dos melhores hiperparâmetros do modelo.
        """
        super().__init__(output_directory, models_directory, best_params_file_name)
        self.cv_results_file_name = cv_results_file_name

    @abstractmethod
    def save_result(self,
                    validation_result: CommonValResult,
                    cv_results,
                    pre_processing_time: str,
                    feature_selection_time: str,
                    search_time: str,
                    validation_time: str,
                    scoring: str,
                    features: list[str],
                    scaler: StandardScaler | None):
        """
        Função que salva todas as informações relevantes ao histórico.

        :param validation_result: Objeto com os dados da validação do melhor modelo encontrado.
        :param cv_results: Dicionário obtido da implementação de busca de parâmetros com as combinações testadas.
        :param pre_processing_time: Tempo que levou o pré-processamento dos dados.
        :param feature_selection_time: Tempo que levou a seleção das melhores features.
        :param search_time: Tempo que levou o processamento de busca de parâmetros.
        :param validation_time: Tempo que levou o processamento de validação do melhor modelo.
        :param scoring: Métrica de validação utilizada.
        :param features: Features selecionadas pela implementação.
        :param scaler: Implementação opcional utilizada para escalar os dados
        """

    def get_dictionary_from_cv_results_json(self, index: int):
        """
        Função que retorna um dicionário com as combinações de valores de parâmetros testadas pela implementação de busca.

        Os dados são recuperados do arquivo JSON e retornados como dicionário.
        """

        return self._get_dictionary_from_json(index, self.cv_results_file_name)

    def _save_model(self, model):
        """
        Salva o modelo treinado utilizando pickle.

        O modelo é salvo em um arquivo .pkl no diretório específico para modelos, com um nome baseado no tamanho do
        histórico atual.

        :param model: O modelo a ser salvo.
        """
        history_len = self.get_history_len()
        output_path = os.path.join(self.models_directory, f"model_{history_len}.pkl")

        with open(output_path, 'wb') as file:
            pickle.dump(model, file)

    def get_saved_model(self, version: int):
        """
        Retorna um modelo salvo a partir da versão.

        :param version: O índice do modelo a ser recuperado.

        :raises FileNotFoundError: Se o modelo não for encontrado.
        """
        output_path = os.path.join(self.models_directory, f"model_{version}.pkl")

        if not os.path.exists(output_path):
            raise FileNotFoundError(
                f"O modelo versão {version} não foi encontrado no diretório {self.models_directory}.")

        with open(output_path, 'rb') as f:
            return pickle.load(f)
