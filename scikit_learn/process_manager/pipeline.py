from typing import Any

from scikit_learn.features_search.common_searcher import CommonFeaturesSearcher
from scikit_learn.hiper_params_search.common_searcher import CommonHiperParamsSearcher
from scikit_learn.history_manager.common import HistoryManager
from scikit_learn.validator.common_validator import Result, BaseValidator


class Pipeline:
    """
    Representa um pipeline de machine learning que combina estimadores,
    busca de hiperparâmetros, seleção de características e validação de modelos.

    Esta classe fornece uma estrutura para integrar todos os componentes
    necessários para a modelagem e validação, permitindo uma execução fluida
    de processos de machine learning.

    :param estimator: O estimador de machine learning a ser utilizado no pipeline.
    :param params: Parâmetros a serem utilizados na busca de hiperparâmetros.
    :param feature_searcher: Objeto responsável pela seleção de features. Pode ser None caso as features possam ser
    definidas sem a necessidade de um algorítmo.
    :param params_searcher: Objeto responsável pela busca de hiperparâmetros.
    :param history_manager: Gerenciador de histórico para armazenar resultados da validação.
    :param validator: Validador que executa a validação do modelo.
    """

    def __init__(self,
                 estimator,
                 params,
                 feature_searcher: CommonFeaturesSearcher | None,
                 params_searcher: CommonHiperParamsSearcher,
                 history_manager: HistoryManager[Result],
                 validator: BaseValidator):
        """
        Inicializa o Pipeline com os componentes fornecidos.

        :param estimator: O estimador a ser utilizado no pipeline.
        :param params: Parâmetros para a busca de hiperparâmetros.
        :param feature_searcher: Objeto que realiza a seleção de características.
        :param params_searcher: Objeto que realiza a busca de hiperparâmetros.
        :param history_manager: Gerenciador de histórico para salvar resultados.
        :param validator: Validador para a validação do modelo.
        """
        self.estimator = estimator
        self.params = params
        self.feature_searcher = feature_searcher
        self.params_searcher = params_searcher
        self.history_manager = history_manager
        self.validator = validator

    def get_dict_pipeline_data(self) -> dict[str, Any]:
        """
        Retorna um dicionário com informações sobre o pipeline, incluindo
        tipos dos componentes utilizados.

        :return: Dicionário contendo os nomes dos tipos de cada componente
                 do pipeline.
        """
        return {
            'estimator': type(self.estimator).__name__,
            'feature_searcher': type(self.feature_searcher).__name__,
            'params_searcher': type(self.params_searcher).__name__,
            'validator': type(self.validator).__name__,
            'history_manager': type(self.history_manager).__name__
        }
