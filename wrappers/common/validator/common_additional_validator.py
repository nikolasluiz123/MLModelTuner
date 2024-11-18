import pandas as pd
import seaborn as sns

from abc import abstractmethod
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate

from wrappers.common.data_pre_processor.common_data_pre_processor import CommonDataPreProcessor


class CommonClassifierAdditionalValidator:
    """
    Implementação que contém tudo que é comum na validação adicional de um modelo de machine learning para classificação.

    O processo de validação no geral é composto por algo fornecido pela biblioteca específica, mas, como uma espécie de
    complemento, é recomendado que sejam feitas algumas validações adicionais com o melhor modelo encontrado.
    """

    def __init__(self,
                 data_pre_processor: CommonDataPreProcessor,
                 confusion_matrix_file_name: str,
                 show_graphics: bool = True,
                 validate_with_train_data: bool = False):
        """
        :param data_pre_processor: Implementação de :class:`wrappers.common.data_pre_processor.common_data_pre_processor.CommonDataPreProcessor`
        para pré-processar os dados para realizar a validação do modelo.

        :param confusion_matrix_file_name: Nome utilizado no arquivo svg da matriz de confusão

        :param show_graphics: Flag que indica se deve ou não serem exibidos os gráficos gerados nessa validação adicional.
                              Por padrão todos os gráficos são salvos como imagem, isso torna possivel não exibir em tempo de execução.

        :param validate_with_train_data: Flag que indica se deve ser utilizado dados de treino ou validação para o processo
                                         da validação adicional
        """
        self.data_pre_processor = data_pre_processor
        self.confusion_matrix_file_name = confusion_matrix_file_name
        self.show_graphics = show_graphics
        self.validate_with_train_data = validate_with_train_data

    @abstractmethod
    def validate(self):
        """
        Função principal da implementação, basicamente é nessa função que devem ser implementados os processos necessários
        para realizar a validação do modelo, utilizando as funções implementadas previamente para auxiliar.
        """

    @staticmethod
    def _show_classification_report(predicted_classes: list, true_classes: list):
        """
        Função que utiliza classification_report do scikit-learn para exibir informações sobre a previsão realizada
        de forma tabular que pode auxiliar na avaliação do modelo.

        :param predicted_classes: Lista das classes previstas pelo modelo avaliado
        :param true_classes: Lista das classes reais do conjunto de dados que o modelo realizou a previsão
        """

        report = classification_report(true_classes, predicted_classes, output_dict=True)
        df_report = pd.DataFrame(report).transpose()

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

        plt.savefig(f'{self.confusion_matrix_file_name}.svg', format='svg')

        if self.show_graphics:
            plt.show()
