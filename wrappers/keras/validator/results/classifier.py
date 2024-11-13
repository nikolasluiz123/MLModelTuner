from typing import Any

from wrappers.keras.validator.results.common import KerasValidationResult


class KerasClassifierValidationResult(KerasValidationResult):

    def append_data(self, pipeline_infos: dict[str, Any]) -> dict[str, Any]:
        pipeline_infos['accuracy'] = self.history.history['accuracy']
        pipeline_infos['val_accuracy'] = self.history.history['val_accuracy']
        pipeline_infos['loss'] = self.history.history['loss']
        pipeline_infos['val_loss'] = self.history.history['val_loss']

        return pipeline_infos

