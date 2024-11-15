import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns

from keras_tuner import Hyperband
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, KFold
from tabulate import tabulate

from wrappers.keras.config.configurators import HyperBandConfig, SearchConfig, FinalFitConfig
from wrappers.keras.validator.common_cross_validator import KerasCrossValidator
from wrappers.keras.validator.results.classifier import KerasClassifierValidationResult
from wrappers.keras.validator.results.common import KerasValidationResult


class ClassifierKerasCrossValidator(KerasCrossValidator):

    def _on_execute(self,
                    train_data,
                    validation_data,
                    model,
                    project_name: str,
                    hyper_band_config: HyperBandConfig,
                    search_config: SearchConfig,
                    final_fit_config: FinalFitConfig) -> KerasValidationResult:
        tuner = Hyperband(
            model,
            objective=hyper_band_config.objective,
            factor=hyper_band_config.factor,
            directory=hyper_band_config.directory,
            project_name=project_name,
            max_epochs=hyper_band_config.max_epochs,
        )

        tuner.search(
            train_data,
            epochs=search_config.epochs,
            validation_data=validation_data,
            batch_size=search_config.batch_size,
            verbose=search_config.log_level,
            callbacks=search_config.callbacks,
        )

        best_hyperparams = tuner.get_best_hyperparameters(num_trials=1)[0]
        model_instance = model.build(best_hyperparams)

        history = self.__execute_final_fit(model_instance=model_instance,
                                           train_data=train_data,
                                           validation_data=validation_data,
                                           final_fit_config=final_fit_config)

        history_dict = {
            'mean_accuracy': round(np.mean(history.history['accuracy']), 2),
            'standard_deviation_accuracy': round(np.std(history.history['accuracy']), 2),

            'mean_val_accuracy': round(np.mean(history.history['val_accuracy']), 2),
            'standard_deviation_val_accuracy': round(np.mean(history.history['val_accuracy']), 2),

            'mean_loss': round(np.mean(history.history['loss']), 2),
            'standard_deviation_loss': round(np.std(history.history['loss']), 2),

            'mean_val_loss': round(np.mean(history.history['val_loss']), 2),
            'standard_deviation_val_loss': round(np.std(history.history['val_loss']), 2),
        }

        return KerasClassifierValidationResult(model_instance, history_dict)

    def __execute_final_fit(self,
                            model_instance,
                            train_data,
                            validation_data,
                            final_fit_config: FinalFitConfig):
        return model_instance.fit(
            train_data,
            epochs=final_fit_config.epochs,
            batch_size=final_fit_config.batch_size,
            verbose=final_fit_config.log_level,
            validation_data=validation_data,
            callbacks=final_fit_config.callbacks,
        )

    def __process_cross_validation(self, data, labels, tuner: Hyperband, search_config: SearchConfig):
        fold = self.__get_fold_implementation(search_config)

        for train_index, validation_index in fold.split(data, labels):
            train_fold_data, validation_fold_data = data[train_index], data[validation_index]
            train_fold_labels, validation_fold_labels = labels[train_index], labels[validation_index]

            train_fold = self.__get_fold_dataset(train_fold_data, train_fold_labels, search_config)
            validation_fold = self.__get_fold_dataset(validation_fold_data, validation_fold_labels, search_config)

            tuner.search(
                train_fold,
                epochs=search_config.epochs,
                validation_data=validation_fold,
                batch_size=search_config.batch_size,
                verbose=search_config.log_level,
                callbacks=search_config.callbacks,
            )

    def __get_fold_implementation(self, search_config: SearchConfig):
        if search_config.stratified:
            fold = StratifiedKFold(n_splits=search_config.folds, shuffle=True)
        else:
            fold = KFold(n_splits=search_config.folds, shuffle=True)
        return fold

    def __get_fold_dataset(self, data_fold, label_fold, search_config: SearchConfig):
        return (tf.data.Dataset.from_tensor_slices((data_fold, label_fold))
                .batch(search_config.batch_size)
                .prefetch(buffer_size=tf.data.AUTOTUNE))

    def __get_tuple_data_labels(self, train_data) -> tuple:
        data, labels = [], []

        for image, label in train_data:
            data.append(image.numpy())
            labels.append(label.numpy())

        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)

        return data, labels


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
