import time

import numpy as np
from keras.src.callbacks import Callback

from wrappers.common.validator.common_validator import CommonValidator
from wrappers.keras.validator.results.classifier_validation_result import KerasClassifierValidationResult


class KerasBasicClassifierValidator(CommonValidator):

    def __init__(self,
                 epochs: int,
                 batch_size: int,
                 callbacks: list[Callback],
                 log_level: int = 0):
        super().__init__(log_level)
        self.epochs = epochs
        self.batch_size = batch_size
        self.callbacks = callbacks

    def validate(self, model_instance, train_data, validation_data) -> KerasClassifierValidationResult:
        self.start_validation_best_model_time = time.time()

        history = model_instance.fit(
            train_data,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.log_level,
            callbacks=self.callbacks,
        )

        history_dict = {
            'mean_accuracy': round(np.mean(history.history['accuracy']), 2),
            'standard_deviation_accuracy': round(np.std(history.history['accuracy']), 2),

            'mean_val_accuracy': round(np.mean(history.history['val_accuracy']), 2),
            'standard_deviation_val_accuracy': round(np.mean(history.history['val_accuracy']), 2),

            'mean_loss': round(np.mean(history.history['loss']), 2),
            'standard_deviation_loss': round(np.std(history.history['loss']), 2),

            'mean_val_loss': round(np.mean(history.history['val_loss']), 2),
            'standard_deviation_val_loss': round(np.std(history.history['val_loss']), 2),
        }

        self.end_validation_best_model_time = time.time()

        return KerasClassifierValidationResult(model_instance, history_dict)