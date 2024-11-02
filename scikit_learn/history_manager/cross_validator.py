from scikit_learn.history_manager.common import HistoryManager
from scikit_learn.validator.results.cross_validation import CrossValidationResult


class CrossValidatorHistoryManager(HistoryManager[CrossValidationResult]):
    """
    Classe que gerencia o histórico de validação cruzada dos modelos.

    Esta classe estende a classe HistoryManager para implementar o armazenamento específico
    dos resultados de validação cruzada. Os resultados são salvos em formato JSON e o modelo treinado
    é armazenado usando pickle, permitindo o uso posterior do modelo.

    :param output_directory: Diretório onde o histórico será armazenado.
    :param models_directory: Diretório onde os modelos treinados serão armazenados.
    :param params_file_name: Nome do arquivo JSON onde os resultados serão salvos.
    """

    def __init__(self, output_directory: str, models_directory: str, params_file_name: str):
        """
        Inicializa o CrossValidatorHistoryManager.

        :param output_directory: Diretório onde o histórico será armazenado.
        :param models_directory: Diretório onde os modelos treinados serão armazenados.
        :param params_file_name: Nome do arquivo JSON onde os resultados serão salvos.
        """
        super().__init__(output_directory, models_directory, params_file_name)

    def save_result(self,
                    classifier_result: CrossValidationResult,
                    feature_selection_time: str,
                    search_time: str,
                    validation_time: str,
                    scoring: str,
                    features: list[str]):
        """
        Salva os resultados da validação cruzada em um arquivo JSON e o modelo treinado em um arquivo pickle.

        :param classifier_result: Objeto CrossValidationResult contendo os dados da validação.
        :param feature_selection_time: Implementação de seleção de features utilizada.
        :param search_time: Tempo que demorou o processamento de busca de parâmetros.
        :param validation_time: Tempo que demorou o processamento de validação do modelo.
        :param scoring: Métrica de validação utilizada.
        :param features: Lista de features selecionadas pela implementação.
        """
        dictionary = {
            'estimator': type(classifier_result.estimator).__name__,
            'mean': classifier_result.mean,
            'standard_deviation': classifier_result.standard_deviation,
            'median': classifier_result.median,
            'variance': classifier_result.variance,
            'standard_error': classifier_result.standard_error,
            'min_max_score': classifier_result.min_max_score,
            'estimator_params': classifier_result.estimator.get_params(),
            'scoring': scoring,
            'features': ", ".join(features),
            'feature_selection_time': feature_selection_time,
            'search_time': search_time,
            'validation_time': validation_time
        }

        self._create_output_dir()  # Garante que os diretórios necessários existam.
        self._save_dictionary_in_json(dictionary)  # Salva o resultado no arquivo JSON.
        self._save_model(classifier_result.estimator)  # Salva o modelo treinado.

    def load_validation_result_from_history(self, index: int = -1) -> CrossValidationResult:
        """
        Carrega um resultado de validação cruzada do histórico a partir de um índice especificado.

        :param index: Índice do resultado a ser carregado. Se -1, o último resultado é retornado.
        :return: Um objeto CrossValidationResult com os dados carregados do histórico.
        :raises IndexError: Se o índice estiver fora dos limites do histórico.
        """
        result_dict = self.get_dictionary_from_json(index)  # Obtém o dicionário do resultado no índice especificado.

        return CrossValidationResult(
            mean=result_dict['mean'],
            standard_deviation=result_dict['standard_deviation'],
            median=result_dict['median'],
            variance=result_dict['variance'],
            standard_error=result_dict['standard_error'],
            min_max_score=result_dict['min_max_score'],
            scoring=result_dict['scoring'],
            estimator=self.get_saved_model(self._get_history_len()),  # Carrega o modelo salvo.
        )
