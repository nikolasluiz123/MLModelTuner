import pandas as pd
import seaborn as sns

from abc import abstractmethod
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate

from wrappers.common.data_pre_processor.common_data_pre_processor import CommonDataPreProcessor


class CommonClassifierAdditionalValidator:

    def __init__(self, data_pre_processor: CommonDataPreProcessor, show_graphics: bool = True):
        self.data_pre_processor = data_pre_processor
        self.show_graphics = show_graphics

    @abstractmethod
    def validate(self):
        ...

    @staticmethod
    def _show_classification_report(predicted_classes, true_classes):
        report = classification_report(true_classes, predicted_classes, output_dict=True)
        df_report = pd.DataFrame(report).transpose()

        print()
        print('Relatório de Classificação:\n')
        print(tabulate(df_report, headers='keys', tablefmt="fancy_grid"))

    def _show_confusion_matrix(self, predicted_classes, true_classes, all_classes_names):
        conf_matrix = confusion_matrix(true_classes, predicted_classes, labels=all_classes_names)
        plt.figure(figsize=(16, 9))
        sns.heatmap(conf_matrix,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    xticklabels=all_classes_names,
                    yticklabels=all_classes_names)

        plt.xlabel("Classes Previstas")
        plt.ylabel("Classes Reais")
        plt.title("Matriz de Confusão")

        plt.savefig(f'confusion_matrix.svg', format='svg')

        if self.show_graphics:
            plt.show()
