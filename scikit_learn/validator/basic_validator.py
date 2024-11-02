import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from scikit_learn.validator.common_validator import BaseValidator, Result


class ClassifierAdditionalValidator:
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
                 data_x,
                 data_y,
                 random_state=42):
        """
        Inicializa um novo validador adicional de classificador.

        :param estimator: O modelo de classificador a ser validado.
        :param data_x: Conjunto de dados de entrada (features).
        :param data_y: Conjunto de dados de saída (rótulos).
        :param random_state: Semente para divisão dos dados. Padrão é 42.
        """
        self.estimator = estimator
        self.data_x = data_x
        self.data_y = data_y
        self.random_state = random_state

    def validate(self):
        """
        Realiza a validação do classificador, dividindo os dados em conjuntos de treinamento e teste,
        ajustando o modelo e exibindo o relatório de classificação e a matriz de confusão.
        """
        x_train, x_test, y_train, y_test = train_test_split(self.data_x,
                                                            self.data_y,
                                                            test_size=0.2,
                                                            random_state=self.random_state)

        self.estimator.fit(x_train, y_train)
        y_pred = self.estimator.predict(x_test)

        self.__show_classification_report(y_test, y_pred)
        self.__show_confusion_matrix(y_test, y_pred)

    def __show_confusion_matrix(self, y_test, y_pred):
        """
        Exibe a matriz de confusão para os rótulos reais e previstos.

        :param y_test: Rótulos reais do conjunto de teste.
        :param y_pred: Rótulos previstos pelo classificador.
        """
        matrix = confusion_matrix(y_test, y_pred)

        classes = np.unique(np.concatenate([y_test, y_pred]))
        class_labels = [f"Classe {cls}" for cls in classes]

        df_cm = pd.DataFrame(matrix, index=class_labels, columns=class_labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=True)
        plt.title("Matriz de Confusão")
        plt.ylabel("Classe Real")
        plt.xlabel("Classe Prevista")
        plt.show()

    def __show_classification_report(self, y_test, y_pred):
        """
        Exibe o relatório de classificação em formato tabular.

        :param y_test: Rótulos reais do conjunto de teste.
        :param y_pred: Rótulos previstos pelo classificador.
        """
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()

        print()
        print('Relatório de Classificação:\n')
        print(tabulate(df_report, headers='keys', tablefmt="fancy_grid"))
