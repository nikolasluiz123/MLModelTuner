from keras_tuner import HyperModel

from wrappers.common.data_pre_processor.common_data_pre_processor import CommonDataPreProcessor
from wrappers.common.process_manager.common_pipeline import CommonPipeline
from wrappers.keras.history_manager.common_history_manager import KerasHistoryManager
from wrappers.keras.hyper_params_search.common_hyper_params_searcher import KerasCommonHyperParamsSearcher
from wrappers.keras.validator.basic_classifier_validator import KerasBasicClassifierValidator


class KerasPipeline(CommonPipeline):

    def __init__(self,
                 model: HyperModel,
                 data_pre_processor: CommonDataPreProcessor,
                 params_searcher: KerasCommonHyperParamsSearcher,
                 validator: KerasBasicClassifierValidator,
                 history_manager: KerasHistoryManager):
        self.model = model
        self.data_pre_processor = data_pre_processor
        self.params_searcher = params_searcher
        self.validator = validator
        self.history_manager = history_manager

    def get_dict_pipeline_data(self) -> dict[str, str]:
        return {
            'model': type(self.model).__name__,
            'data_pre_processor': type(self.data_pre_processor).__name__,
            'params_searcher': type(self.params_searcher).__name__,
            'searcher_objective': self.params_searcher.objective,
            'searcher_epochs': self.params_searcher.epochs,
            'searcher_batch_size': self.params_searcher.batch_size,
            'project_name': self.params_searcher.project_name,
            'validator': type(self.validator).__name__,
            'validator_epochs': self.validator.epochs,
            'validator_batch_size': self.validator.batch_size
        }

    def get_execution_times(self):
        pre_processing_time = self.data_pre_processor.end_pre_processing - self.data_pre_processor.start_pre_processing
        params_search_time = self.params_searcher.end_search_parameter_time - self.params_searcher.start_search_parameter_time
        validation_time = self.validator.end_validation_best_model_time - self.validator.start_validation_best_model_time

        return pre_processing_time, params_search_time, validation_time