import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from keras_tuner import Hyperband
from sklearn.model_selection import StratifiedKFold, KFold

from wrappers.keras.process_manager.pipeline import KerasPipeline
from wrappers.keras.validator.common_cross_validator import KerasCrossValidator
from wrappers.keras.validator.results.common import KerasValidationResult


class ClassifierKerasCrossValidator(KerasCrossValidator):

    def _on_execute(self, train_data, validation_data, pipeline: KerasPipeline, history_index: int = None) -> KerasValidationResult:
        data, labels = self.__get_tuple_data_labels(train_data)

        tuner = Hyperband(
            pipeline.model,
            objective=pipeline.hyper_band_config.objective,
            factor=pipeline.hyper_band_config.factor,
            directory=pipeline.hyper_band_config.directory,
            project_name=self.__get_project_name(pipeline, history_index),
            max_epochs=pipeline.hyper_band_config.max_epochs,
        )

        self.__process_cross_validation(data, labels, tuner, pipeline)

        best_hyperparams = tuner.get_best_hyperparameters(num_trials=1)[0]
        model_instance = pipeline.model.build(best_hyperparams)

        history = self.__execute_final_fit(model_instance, train_data, validation_data, pipeline, history_index)

        return KerasValidationResult(model_instance, history)

    def __execute_final_fit(self, model_instance, train_data, validation_data, pipeline: KerasPipeline, history_index: int):
        if history_index is None:
            return model_instance.fit(
                train_data,
                epochs=pipeline.final_fit_config.epochs,
                batch_size=pipeline.final_fit_config.batch_size,
                verbose=pipeline.final_fit_config.log_level,
                validation_data=validation_data,
            )
        else:
            return pipeline.history_manager.get_history_from_best_model_executions(history_index)

    def __get_project_name(self, pipeline: KerasPipeline, history_index: int):
        if history_index is not None:
            directory = pipeline.hyper_band_config.directory
            projects = os.listdir(directory)

            project_names = [p for p in projects if os.path.isdir(os.path.join(directory, p))]

            if history_index >= len(project_names):
                raise IndexError(f'Não foi possível recuperar a pasta do histórico de execuções com o índice {history_index}')

            return project_names[history_index]
        else:
            date_time_now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            return f'{pipeline.hyper_band_config.project_name}_{date_time_now}'

    def __process_cross_validation(self, data, labels, tuner, pipeline: KerasPipeline):
        fold = self.__get_fold_implementation(pipeline)

        for train_index, validation_index in fold.split(data):
            train_fold_data, validation_fold_data = data[train_index], data[validation_index]
            train_fold_labels, validation_fold_labels = labels[train_index], labels[validation_index]

            train_fold = self.__get_fold_dataset(train_fold_data, train_fold_labels, pipeline)
            validation_fold = self.__get_fold_dataset(validation_fold_data, validation_fold_labels, pipeline)

            tuner.search(
                train_fold,
                epochs=pipeline.search_config.epochs,
                validation_data=validation_fold,
                batch_size=pipeline.search_config.batch_size,
                verbose=pipeline.search_config.log_level,
                callbacks=pipeline.search_config.callbacks,
            )

    def __get_fold_implementation(self, pipeline: KerasPipeline):
        if pipeline.search_config.stratified:
            fold = StratifiedKFold(n_splits=pipeline.search_config.folds, shuffle=True)
        else:
            fold = KFold(n_splits=pipeline.search_config.folds, shuffle=True)
        return fold

    def __get_fold_dataset(self, data_fold, label_fold, pipeline: KerasPipeline):
        return (tf.data.Dataset.from_tensor_slices((data_fold, label_fold))
                .batch(pipeline.search_config.batch_size)
                .prefetch(buffer_size=tf.data.AUTOTUNE))

    def __get_tuple_data_labels(self, train_data) -> tuple:
        data, labels = [], []

        for image, label in train_data:
            data.append(image.numpy())
            labels.append(label.numpy())

        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)

        return data, labels
