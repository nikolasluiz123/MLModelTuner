from typing import Any

from sklearn.preprocessing import StandardScaler

from wrappers.common.data_pre_processor.common_data_pre_processor import CommonDataPreProcessor
from wrappers.common.history_manager.common_history_manager import CommonValResult
from wrappers.common.process_manager.common_pipeline import CommonPipeline
from wrappers.scikit_learn.features_search.common_feature_searcher import ScikitLearnCommonFeaturesSearcher
from wrappers.scikit_learn.hyper_params_search.common_hyper_params_searcher import ScikitLearnCommonHyperParamsSearcher
from wrappers.scikit_learn.history_manager.common_history_manager import ScikitLearnCommonHistoryManager
from wrappers.scikit_learn.validator.common_validator import ScikitLearnCommonValidator


class ScikitLearnPipeline(CommonPipeline):
    """
    Implementação de pipeline específica para buscar o melhor modelo do scikit-learn
    """

    def __init__(self,
                 estimator,
                 params,
                 data_pre_processor: CommonDataPreProcessor,
                 feature_searcher: ScikitLearnCommonFeaturesSearcher | None,
                 params_searcher: ScikitLearnCommonHyperParamsSearcher,
                 history_manager: ScikitLearnCommonHistoryManager[CommonValResult],
                 validator: ScikitLearnCommonValidator,
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
        self.data_pre_processor = data_pre_processor
        self.feature_searcher = feature_searcher
        self.params_searcher = params_searcher
        self.history_manager = history_manager
        self.validator = validator

    def get_dictionary_pipeline_data(self) -> dict[str, Any]:
        return {
            'estimator': type(self.estimator).__name__,
            'scaler': type(self.scaler).__name__,
            'data_pre_processor': type(self.data_pre_processor).__name__,
            'feature_searcher': type(self.feature_searcher).__name__,
            'params_searcher': type(self.params_searcher).__name__,
            'validator': type(self.validator).__name__,
            'history_manager': type(self.history_manager).__name__
        }

    def get_execution_times(self):
        if self.feature_searcher is not None:
            features_selection_time = self.feature_searcher.end_search_features_time - self.feature_searcher.start_search_features_time
        else:
            features_selection_time = 0

        pre_processing_time = self.data_pre_processor.end_pre_processing - self.data_pre_processor.start_pre_processing
        params_search_time = self.params_searcher.end_search_parameter_time - self.params_searcher.start_search_parameter_time
        validation_time = self.validator.end_validation_best_model_time - self.validator.start_validation_best_model_time

        return pre_processing_time, features_selection_time, params_search_time, validation_time
