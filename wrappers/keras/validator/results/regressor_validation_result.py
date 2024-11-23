from typing import Any

from wrappers.keras.validator.results.common_validation_result import KerasValidationResult


class KerasRegressorValidationResult(KerasValidationResult):

    def append_data(self, pipeline_infos: dict[str, Any]) -> dict[str, Any]:
        pipeline_infos['mean_absolute_error'] = self.history['mean_absolute_error']
        pipeline_infos['standard_deviation_absolute_error'] = self.history['standard_deviation_absolute_error']

        pipeline_infos['mean_val_absolute_error'] = self.history['mean_val_absolute_error']
        pipeline_infos['standard_deviation_val_absolute_error'] = self.history['standard_deviation_val_absolute_error']

        pipeline_infos['mean_loss'] = self.history['mean_loss']
        pipeline_infos['standard_deviation_loss'] = self.history['standard_deviation_loss']

        pipeline_infos['mean_val_loss'] = self.history['mean_val_loss']
        pipeline_infos['standard_deviation_val_loss'] = self.history['standard_deviation_val_loss']

        return pipeline_infos

