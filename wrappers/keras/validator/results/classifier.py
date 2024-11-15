from typing import Any

from wrappers.keras.validator.results.common import KerasValidationResult


class KerasClassifierValidationResult(KerasValidationResult):

    def append_data(self, pipeline_infos: dict[str, Any]) -> dict[str, Any]:
        pipeline_infos['mean_accuracy'] = self.history['mean_accuracy']
        pipeline_infos['standard_deviation_accuracy'] = self.history['standard_deviation_accuracy']

        pipeline_infos['mean_val_accuracy'] = self.history['mean_val_accuracy']
        pipeline_infos['standard_deviation_val_accuracy'] = self.history['standard_deviation_val_accuracy']

        pipeline_infos['mean_loss'] = self.history['mean_loss']
        pipeline_infos['standard_deviation_loss'] = self.history['standard_deviation_loss']

        pipeline_infos['mean_val_loss'] = self.history['mean_val_loss']
        pipeline_infos['standard_deviation_val_loss'] = self.history['standard_deviation_val_loss']

        return pipeline_infos

