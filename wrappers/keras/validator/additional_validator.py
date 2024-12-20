import numpy as np
from sklearn.preprocessing import MinMaxScaler

from wrappers.common.validator.common_additional_validator import CommonClassifierAdditionalValidator, \
    CommonRegressorAdditionalValidator


class KerasAdditionalClassifierValidator(CommonClassifierAdditionalValidator):
    """
    Implementação para realizar a validação adicional de uma rede neural de classificação.
    """
    def __init__(self,
                 data,
                 model_instance,
                 validation_results_directory: str,
                 prefix_file_names: str,
                 show_graphics: bool = True):
        """
        :param model_instance: Instância do modelo que já passou pelos processos de treino e foi avaliado como o melhor
                               modelo pelos processos comuns genéricos
        """
        super().__init__(data, validation_results_directory, prefix_file_names, show_graphics)
        self.model_instance = model_instance

    def validate(self):
        true_labels = []

        for _, label in self.data:
            true_labels.extend(label.numpy())

        predictions = self.model_instance.predict(self.data)
        predicted_classes = np.argmax(predictions, axis=1)

        classes_names = sorted(set(self.data.class_names))

        predicted_class_names = [classes_names[i] for i in predicted_classes]
        true_class_names = [classes_names[i] for i in true_labels]

        self._show_classification_report(predicted_class_names, true_class_names)
        self._show_confusion_matrix(predicted_class_names, true_class_names, classes_names)

class KerasAdditionalRegressorValidator(CommonRegressorAdditionalValidator):
    def __init__(self,
                 data,
                 model_instance,
                 validation_results_directory: str,
                 prefix_file_names: str,
                 scaler: MinMaxScaler = None,
                 show_graphics: bool = True):
        super().__init__(data, validation_results_directory, prefix_file_names, show_graphics)
        self.model_instance = model_instance
        self.scaler = scaler

    def validate(self):
        predictions_scaled = self.model_instance.predict(self.data[0])
        predictions = self.scaler.inverse_transform(predictions_scaled)
        y_test_original = self.scaler.inverse_transform(self.data[1])

        self._show_regression_report(predictions, y_test_original)
        self._show_regression_graph(predictions, y_test_original)