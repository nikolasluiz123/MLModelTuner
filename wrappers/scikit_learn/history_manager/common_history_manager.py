import os
import pickle
from abc import abstractmethod, ABC

from sklearn.preprocessing import StandardScaler

from wrappers.common.history_manager.common_history_manager import CommonHistoryManager, CommonValResult


class ScikitLearnCommonHistoryManager(CommonHistoryManager[CommonValResult], ABC):
    """
    Classe utilizada para gerenciamento do histórico das execuções.

    Os resultados das execuções são salvos em um diretório especificado e em um arquivo JSON com o nome desejado. A
    estrutura do JSON é uma lista e os campos inseridos nele dependem do objeto `Result` e de quais campos desse objeto
    for julgado relevante manter no histórico.

    Além de salvar os dados do objeto de resultado da validação, também salvamos o modelo em si, utilizando o pickle. Dessa
    forma, é possível reutilizar o modelo treinado e validado para algum fim específico.
    """

    def __init__(self, output_directory: str, models_directory: str, best_params_file_name: str, cv_results_file_name: str):
        """
        Inicializa o HistoryManager com os diretórios e nome de arquivo apropriados.

        :param output_directory: Diretório de histórico que vai armazenar o arquivo JSON e os modelos.
        :param models_directory: Diretório específico para os modelos.
        :param best_params_file_name: Nome do arquivo JSON o qual conterá os parâmetros e resultados.
        :param cv_results_file_name: Nome do arquivo JSON que conterá as combinações de valores dos parâmetros da execução.
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
        Função que deve ser implementada para salvar os dados do objeto `Result` no arquivo JSON.

        :param validation_result: Objeto com os dados da validação do melhor modelo encontrado.
        :param cv_results: Dicionário obtido da implementação de busca de parâmetros com as combinações testadas.
        :param feature_selection_time: Implementação de seleção de features utilizada.
        :param search_time: Tempo que demorou o processamento de busca de parâmetros.
        :param validation_time: Tempo que demorou o processamento de validação do melhor modelo.
        :param scoring: Métrica de validação utilizada.
        :param features: Features selecionadas pela implementação.
        :param scaler: Implementação opcional utilizada para escalar os dados
        """

    def get_dictionary_from_cv_results_json(self, index: int):
        return self._get_dictionary_from_json(index, self.cv_results_file_name)

    def _save_model(self, estimator):
        """
        Salva o modelo treinado utilizando pickle.

        O modelo é salvo em um arquivo .pkl no diretório específico para modelos,
        com um nome baseado no tamanho do histórico atual.

        :param estimator: O modelo a ser salvo.
        """
        history_len = self.get_history_len()
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
