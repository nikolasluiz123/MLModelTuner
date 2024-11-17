import numpy as np
from sklearn.model_selection import train_test_split

from wrappers.common.data_pre_processor.common_data_pre_processor import CommonDataPreProcessor
from wrappers.common.validator.common_additional_validator import CommonClassifierAdditionalValidator


class ScikitLearnClassifierAdditionalValidator(CommonClassifierAdditionalValidator):
    """
    Classe responsável por validar um classificador por meio de métricas de desempenho e visualizações.

    Esta classe permite realizar a validação de um modelo de classificação, incluindo a geração de um
    relatório de classificação e a exibição de uma matriz de confusão.

    :param estimator: O modelo de classificador a ser validado.
    :param data_x: Conjunto de dados de entrada (features) para treinamento e teste.
    :param data_y: Conjunto de dados de saída (rótulos) para treinamento e teste.
    :param random_state: Semente para geração de números aleatórios, utilizada na divisão dos dados.
    """

    def __init__(self,
                 estimator,
                 data_pre_processor: CommonDataPreProcessor,
                 random_state=42,
                 show_graphics: bool = True):
        """
        Inicializa um novo validador adicional de classificador.

        :param estimator: O modelo de classificador a ser validado.
        :param data_x: Conjunto de dados de entrada (features).
        :param data_y: Conjunto de dados de saída (rótulos).
        :param random_state: Semente para divisão dos dados. Padrão é 42.
        """
        super().__init__(data_pre_processor, show_graphics)
        self.estimator = estimator
        self.random_state = random_state

    def validate(self):
        """
        Realiza a validação do classificador, dividindo os dados em conjuntos de treinamento e teste,
        ajustando o modelo e exibindo o relatório de classificação e a matriz de confusão.
        """
        data_x, data_y = self.data_pre_processor.get_data_additional_validation()

        x_train, x_test, y_train, y_test = train_test_split(data_x,
                                                            data_y,
                                                            test_size=0.2,
                                                            random_state=self.random_state)

        self.estimator.fit(x_train, y_train)
        y_pred = self.estimator.predict(x_test)

        self._show_classification_report(y_test, y_pred)
        self._show_confusion_matrix(y_test, y_pred, np.unique(data_y))

