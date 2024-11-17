from wrappers.common.history_manager.common_history_manager import CommonValResult
from wrappers.keras.history_manager.common_history_manager import KerasHistoryManager
from wrappers.keras.validator.results.classifier_validation_result import KerasClassifierValidationResult


class KerasClassifierHistoryManager(KerasHistoryManager[KerasClassifierValidationResult]):

    def load_validation_result_from_history(self, index: int = -1) -> CommonValResult:
        execution_data = self.get_dictionary_from_params_json(index)
        model_instance = self.get_saved_model(self.get_history_len())

        return KerasClassifierValidationResult(model_instance, execution_data['history'])
