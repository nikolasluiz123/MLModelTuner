from wrappers.keras.history_manager.common_history_manager import KerasHistoryManager, Result
from wrappers.keras.validator.results.classifier import KerasClassifierValidationResult


class KerasClassifierHistoryManager(KerasHistoryManager[KerasClassifierValidationResult]):

    def get_validation_result(self, index: int) -> Result:
        execution_data = self._get_dictionary_from_json(self.output_directory, index, self.best_executions_file_name)
        model_instance = self.get_best_model_executions(index)

        return KerasClassifierValidationResult(model_instance, execution_data['history'])