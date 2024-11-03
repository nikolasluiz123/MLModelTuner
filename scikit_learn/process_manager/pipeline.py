from typing import Any

from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

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
    """

    def __init__(self,
                 estimator,
                 params,
                 feature_searcher: CommonFeaturesSearcher | None,
                 params_searcher: CommonHiperParamsSearcher,
                 history_manager: HistoryManager[Result],
                 validator: BaseValidator,
                 scaler: StandardScaler | None = None):
        """
        Inicializa o Pipeline com os componentes fornecidos.

        :param estimator: O estimador a ser utilizado no pipeline.
        :param params: Parâmetros para a busca de hiperparâmetros.
        :param feature_searcher: Objeto que realiza a seleção de características.
        :param params_searcher: Objeto que realiza a busca de hiperparâmetros.
        :param history_manager: Gerenciador de histórico para salvar resultados.
        :param validator: Validador para a validação do modelo.
        :param scaler: Transformer para manipular os dados de alguma maneira, por exemplo, escalar eles. É opcional,
        pois alguns modelos não se beneficiam dessa estratégia.
        """
        self.estimator = estimator
        self.params = params
        self.scaler = scaler
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
            'scaler': type(self.scaler).__name__,
            'feature_searcher': type(self.feature_searcher).__name__,
            'params_searcher': type(self.params_searcher).__name__,
            'validator': type(self.validator).__name__,
            'history_manager': type(self.history_manager).__name__
        }
