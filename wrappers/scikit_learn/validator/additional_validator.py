import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from wrappers.common.validator.common_additional_validator import CommonClassifierAdditionalValidator, \
    CommonRegressorAdditionalValidator


class ScikitLearnClassifierAdditionalValidator(CommonClassifierAdditionalValidator):
    """
    Implementação de validação adicional para modelos de classificação implementados a partir do scikit-learn.
    """

    def __init__(self,
                 data,
                 estimator,
                 validation_results_directory: str,
                 prefix_file_names: str,
                 random_state=42,
                 show_graphics: bool = True,
                 label_encoder: LabelEncoder = None):
        """
        :param estimator: Modelo do scikit-learn que será avaliado
        :param random_state: Seed utilizada no split dos dados. É necessária, pois a validação adicional ocorre fora do
                              ProcessManager.
        :param label_encoder: Implementação que foi utilizada para transformar valores string categóricos em números. Nesse
                             caso, ao realizar a validação adicional, pode ser desejado visualizar os valores string.
        """

        super().__init__(data, validation_results_directory, prefix_file_names, show_graphics)
        self.estimator = estimator
        self.random_state = random_state
        self.label_encoder = label_encoder

    def validate(self):
        y_test = self.data[1]
        y_pred = self.estimator.predict(self.data[0])

        if self.label_encoder is not None:
            y_test = self.label_encoder.inverse_transform(y_test)
            y_pred = self.label_encoder.inverse_transform(y_pred)
            all_classes_names = list(self.label_encoder.inverse_transform(list(np.unique(self.data[1]))))
        else:
            all_classes_names = list(np.unique(self.data[1]))

        self._show_classification_report(y_test, y_pred)
        self._show_confusion_matrix(y_test, y_pred, all_classes_names)

class ScikitLearnRegressorAdditionalValidator(CommonRegressorAdditionalValidator):

    def __init__(self,
                 data,
                 estimator,
                 validation_results_directory: str,
                 prefix_file_names: str,
                 random_state=42,
                 show_graphics: bool = True):
        super().__init__(data, validation_results_directory, prefix_file_names, show_graphics)
        self.estimator = estimator
        self.random_state = random_state

    def validate(self):
        x_train, x_test, y_train, y_test = train_test_split(self.data[0],
                                                            self.data[1],
                                                            test_size=0.2,
                                                            random_state=self.random_state)

        self.estimator.fit(x_train, y_train)
        y_pred = self.estimator.predict(x_test)

        self._show_regression_report(y_test, y_pred)
        self._show_regression_graph(y_test, y_pred)