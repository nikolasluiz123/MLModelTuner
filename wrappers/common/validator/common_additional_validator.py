import os

import numpy as np
import pandas as pd
import seaborn as sns

from abc import abstractmethod
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score
)


class CommonClassifierAdditionalValidator:
    """
    Implementação que contém tudo que é comum na validação adicional de um modelo de machine learning para classificação.

    O processo de validação no geral é composto por algo fornecido pela biblioteca específica, mas, como uma espécie de
    complemento, é recomendado que sejam feitas algumas validações adicionais com o melhor modelo encontrado.
    """

    def __init__(self,
                 data,
                 validation_results_directory: str,
                 prefix_file_names: str,
                 show_graphics: bool = True):
        """
        :param data: Dados que serão usados na validação

        :param prefix_file_names: Prefixo utilizado no nome de todos os arquivos que são salvos

        :param validation_results_directory: Diretório onde todos os resultados das validações serão salvas

        :param show_graphics: Flag que indica se deve ou não serem exibidos os gráficos gerados nessa validação adicional.
                              Por padrão todos os gráficos são salvos como imagem, isso torna possivel não exibir em tempo de execução.
        """
        self.data = data
        self.validation_results_directory = validation_results_directory
        self.prefix_file_names = prefix_file_names
        self.show_graphics = show_graphics

        self._create_output_dir()

    def _create_output_dir(self):
        """
        Cria os diretórios de saída para as validações, caso não existam.
        """
        if not os.path.exists(self.validation_results_directory):
            os.makedirs(self.validation_results_directory)

    @abstractmethod
    def validate(self):
        """
        Função principal da implementação, basicamente é nessa função que devem ser implementados os processos necessários
        para realizar a validação do modelo, utilizando as funções implementadas previamente para auxiliar.
        """

    def _show_classification_report(self, predicted_classes: list, true_classes: list):
        """
        Função que utiliza classification_report do scikit-learn para exibir informações sobre a previsão realizada
        de forma tabular que pode auxiliar na avaliação do modelo.

        :param predicted_classes: Lista das classes previstas pelo modelo avaliado
        :param true_classes: Lista das classes reais do conjunto de dados que o modelo realizou a previsão
        """

        report = classification_report(true_classes, predicted_classes, output_dict=True)
        df_report = pd.DataFrame(report).transpose()

        file_name = f'{self.prefix_file_names}_classification_report.csv'
        output_dir = os.path.join(self.validation_results_directory, file_name)

        df_report.to_csv(output_dir)

        print()
        print('Relatório de Classificação:\n')
        print(tabulate(df_report, headers='keys', tablefmt="fancy_grid"))

    def _show_confusion_matrix(self, predicted_classes: list, true_classes: list, all_classes_names: list):
        """
        Função que utiliza confusion_matrix do scikit-learn para exibir um gráfico heatmap e possibilitar visualizar se
        o modelo de classificação está confundindo alguma das classes que você está tendando prever.

        :param predicted_classes: Lista das classes previstas pelo modelo avaliado
        :param true_classes: Lista das classes reais do conjunto de dados que o modelo realizou a previsão
        :param all_classes_names: Lista de todas as classes do conjunto de dados
        """
        conf_matrix = confusion_matrix(true_classes, predicted_classes, labels=all_classes_names)
        plt.figure(figsize=(16, 9))
        sns.heatmap(conf_matrix,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    cbar=False,
                    xticklabels=all_classes_names,
                    yticklabels=all_classes_names)

        plt.xlabel('Classes Previstas')
        plt.ylabel('Classes Reais')
        plt.title('Matriz de Confusão')

        file_name = f'{self.prefix_file_names}_confusion_matrix.svg'
        output_dir = os.path.join(self.validation_results_directory, file_name)
        plt.savefig(output_dir, format='svg')

        if self.show_graphics:
            plt.show()


class CommonRegressorAdditionalValidator:
    """
    Implementação que contém tudo que é comum na validação adicional de um modelo de machine learning para regressão.

    O processo de validação foi inspirado no que é disponibilizado para classificadores, dessa forma, foi implementada
    uma função que reune métricas matemáticas utilizadas para avaliar um modelo de regressão. Além disso, gráficos também
    foram utilizados para demonstração visual.
    """

    def __init__(self,
                 data,
                 validation_results_directory: str,
                 prefix_file_names: str,
                 show_graphics: bool = True):
        self.data = data
        self.validation_results_directory = validation_results_directory
        self.prefix_file_names = prefix_file_names
        self.show_graphics = show_graphics

        self._create_output_dir()

    def _create_output_dir(self):
        if not os.path.exists(self.validation_results_directory):
            os.makedirs(self.validation_results_directory)

    @abstractmethod
    def validate(self):
        """
        Função principal da implementação, basicamente é nessa função que devem ser implementados os processos necessários
        para realizar a validação do modelo, utilizando as funções implementadas previamente para auxiliar.
        """

    def _show_regression_report(self, predicted_classes: list, true_classes: list):
        """
        Função que cria um relatório com métricas que podem auxiliar no julgamento de um modelo de regressão.

        Esse relatório é exibido como um dataframe tabular, além de ser salvo como csv no diretório desejado para posterior
        reavaliação.

        :param predicted_classes: Lista das classes previstas pelo modelo avaliado
        :param true_classes: Lista das classes reais do conjunto de dados que o modelo realizou a previsão
        """

        report = {
            "Mean Absolute Error (MAE)": mean_absolute_error(true_classes, predicted_classes),
            "Mean Squared Error (MSE)": mean_squared_error(true_classes, predicted_classes),
            "Root Mean Squared Error (RMSE)": np.sqrt(mean_squared_error(true_classes, predicted_classes)),
            "R² Score": r2_score(true_classes, predicted_classes),
            "Explained Variance": explained_variance_score(true_classes, predicted_classes),
        }

        df_report = pd.DataFrame.from_dict(report, orient='index', columns=['Valor'])
        df_report.index.name = 'Métrica'
        df_report.reset_index(inplace=True)

        file_name = f'{self.prefix_file_names}_regression_report.csv'
        output_dir = os.path.join(self.validation_results_directory, file_name)

        df_report.to_csv(output_dir)

        print()
        print('Relatório de Regressão:\n')
        print(tabulate(df_report, headers='keys', tablefmt="fancy_grid"))

    def _show_regression_graph(self, predicted_classes: list, true_classes: list):
        """
        Função utilizada para gerar um gráfico onde os dados reais e as predições são sobrepostas em forma de linha,
        dessa maneira é possível verificar o quão bem o modelo está se ajustando aos dados.

        :param predicted_classes: Lista das classes previstas pelo modelo avaliado
        :param true_classes: Lista das classes reais do conjunto de dados que o modelo realizou a previsão
        """
        plt.figure(figsize=(12, 6))
        plt.plot(true_classes, label="Valores Reais")
        plt.plot(predicted_classes, label="Previsões")
        plt.legend()
        plt.title("Regressão: Real vs Previsto")

        file_name = f'{self.prefix_file_names}_regression_result.svg'
        output_dir = os.path.join(self.validation_results_directory, file_name)
        plt.savefig(output_dir, format='svg')

        if self.show_graphics:
            plt.show()
