from typing import Any

from scikit_learn.features_search.common_searcher import CommonFeaturesSearcher
from scikit_learn.hiper_params_search.common_searcher import CommonHiperParamsSearcher
from scikit_learn.history_manager.common import HistoryManager
from scikit_learn.validator.common_validator import Result, BaseValidator


class Pipeline:

    def __init__(self,
                 estimator,
                 params,
                 feature_searcher: CommonFeaturesSearcher,
                 params_searcher: CommonHiperParamsSearcher,
                 history_manager: HistoryManager[Result],
                 validator: BaseValidator):
        self.estimator = estimator
        self.params = params
        self.feature_searcher = feature_searcher
        self.params_searcher = params_searcher
        self.history_manager = history_manager
        self.validator = validator

    def get_dict_pipeline_data(self) -> dict[str, Any]:
        return {
            'estimator': type(self.estimator).__name__,
            'feature_searcher': type(self.feature_searcher).__name__,
            'params_searcher': type(self.params_searcher).__name__,
            'validator': type(self.validator).__name__,
            'history_manager': type(self.history_manager).__name__
        }
