import numpy as np
from sklearn.model_selection import train_test_split

from wrappers.common.data_pre_processor.common_data_pre_processor import CommonDataPreProcessor
from wrappers.common.validator.common_additional_validator import CommonClassifierAdditionalValidator


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
                 show_graphics: bool = True):
        """
        :param estimator: Modelo do scikit-learn que será avaliado
        :param: random_state: Seed utilizada no split dos dados. É necessária, pois a validação adicional ocorre fora do
                              ProcessManager.
        """

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

        self._show_classification_report(y_test, y_pred)
        self._show_confusion_matrix(y_test, y_pred, list(np.unique(self.data[1])))

