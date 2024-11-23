from wrappers.common.history_manager.common_history_manager import CommonValResult
from wrappers.keras.history_manager.common_history_manager import KerasCommonHistoryManager
from wrappers.keras.validator.results.regressor_validation_result import KerasRegressorValidationResult


class KerasRegressorHistoryManager(KerasCommonHistoryManager[KerasRegressorValidationResult]):

    def load_validation_result_from_history(self, index: int = -1) -> CommonValResult:
        execution_data = self.get_dictionary_from_params_json(index)
        model_instance = self.get_saved_model(self.get_history_len())

        return KerasRegressorValidationResult(model_instance, execution_data['history'])
