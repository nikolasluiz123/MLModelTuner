import time

import numpy as np

from wrappers.keras.validator.common_basic_validator import KerasCommonBasicValidator
from wrappers.keras.validator.results.classifier_validation_result import KerasClassifierValidationResult


class KerasBasicClassifierValidator(KerasCommonBasicValidator[KerasClassifierValidationResult]):
    """
    Implementação para realizar a validação de uma rede neural de classificação.
    """

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
            'mean_accuracy': np.mean(history.history['accuracy']),
            'standard_deviation_accuracy': np.std(history.history['accuracy']),

            'mean_val_accuracy': np.mean(history.history['val_accuracy']),
            'standard_deviation_val_accuracy': np.mean(history.history['val_accuracy']),

            'mean_loss': np.mean(history.history['loss']),
            'standard_deviation_loss': np.std(history.history['loss']),

            'mean_val_loss': np.mean(history.history['val_loss']),
            'standard_deviation_val_loss': np.std(history.history['val_loss']),
        }

        self.end_validation_best_model_time = time.time()

        return KerasClassifierValidationResult(model_instance, history_dict)