import time

import numpy as np

from wrappers.keras.validator.common_basic_validator import KerasCommonBasicValidator
from wrappers.keras.validator.results.regressor_validation_result import KerasRegressorValidationResult


class KerasBasicRegressorValidator(KerasCommonBasicValidator[KerasRegressorValidationResult]):
    """
    Implementação para realizar a validação de uma rede neural de regressão.
    """

    def validate(self, model_instance, train_data, validation_data) -> KerasRegressorValidationResult:
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
            'mean_absolute_error': np.mean(history.history['mean_absolute_error']),
            'standard_deviation_absolute_error': np.std(history.history['mean_absolute_error']),

            'mean_val_absolute_error': np.mean(history.history['val_mean_absolute_error']),
            'standard_deviation_val_absolute_error': np.mean(history.history['val_mean_absolute_error']),

            'mean_loss': np.mean(history.history['loss']),
            'standard_deviation_loss': np.std(history.history['loss']),

            'mean_val_loss': np.mean(history.history['val_loss']),
            'standard_deviation_val_loss': np.std(history.history['val_loss']),
        }

        self.end_validation_best_model_time = time.time()

        return KerasRegressorValidationResult(model_instance, history_dict)