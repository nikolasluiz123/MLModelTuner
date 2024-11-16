import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tabulate import tabulate


class KerasAdditionalClassifierValidator:

    def __init__(self, model_instance, model, history_dict: dict, data):
        self.model_instance = model_instance
        self.model = model
        self.history_dict = history_dict
        self.data = data

    def validate(self, show_graphic: bool = False):
        true_labels = []

        for _, label in self.data:
            true_labels.extend(label.numpy())

        predictions = self.model_instance.predict(self.data)
        predicted_classes = np.argmax(predictions, axis=1)

        classes_names = sorted(set(self.data.class_names))

        predicted_class_names = [classes_names[i] for i in predicted_classes]
        true_class_names = [classes_names[i] for i in true_labels]

        self.__show_classification_report(predicted_class_names, true_class_names)
        self.__show_confusion_matrix(predicted_class_names, true_class_names, classes_names, show_graphic)

    def __show_classification_report(self, predicted_classes, true_labels):
        report = classification_report(true_labels, predicted_classes, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        print()
        print('Relatório de Classificação:\n')
        print(tabulate(df_report, headers='keys', tablefmt="fancy_grid"))

    def __show_confusion_matrix(self, predicted_classes, true_labels, classes_names, show_graphic: bool):
        conf_matrix = confusion_matrix(true_labels, predicted_classes, labels=classes_names)
        plt.figure(figsize=(16, 9))
        sns.heatmap(conf_matrix,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    xticklabels=classes_names,
                    yticklabels=classes_names)

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.xlabel("Classes Previstas")
        plt.ylabel("Classes Reais")
        plt.title("Matriz de Confusão")

        plt.savefig(f'confusion_matrix_{type(self.model).__name__}.svg', format='svg')

        if show_graphic:
            plt.show()
