import numpy as np
import tensorflow as tf
import keras

from keras.src.callbacks import EarlyStopping
from keras_tuner import Hyperband
from sklearn.model_selection import KFold

class HyperBandConfig:

    def __init__(self,
               objective,
               factor,
               max_epochs,
               directory,
               project_name):
        self.objective = objective
        self.factor = factor
        self.max_epochs = max_epochs
        self.directory = directory
        self.project_name = project_name

class SearchConfig:

    def __init__(self,
                 epochs,
                 batch_size,
                 callbacks,
                 folds,
                 log_level):
        self.epochs = epochs
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.folds = folds
        self.log_level = log_level

class FinalFitConfig:

    def __init__(self,
                 epochs,
                 batch_size,
                 log_level):
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_level = log_level


class CrossValidator:

    def __init__(self,
                 train_data,
                 validation_data,
                 hyper_band_config: HyperBandConfig,
                 search_config: SearchConfig,
                 final_fit_config: FinalFitConfig):
        self.train_data = train_data
        self.validation_data = validation_data
        self.hyper_band_config = hyper_band_config
        self.search_config = search_config
        self.final_fit_config = final_fit_config

    def execute(self, model):
        images, labels = [], []

        for image, label in self.train_data:
            images.append(image.numpy())
            labels.append(label.numpy())

        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)

        tuner = Hyperband(
            model,
            objective=self.hyper_band_config.objective,
            factor=self.hyper_band_config.factor,
            directory=self.hyper_band_config.directory,
            project_name=self.hyper_band_config.project_name,
            max_epochs=self.hyper_band_config.max_epochs,
        )

        fold = KFold(n_splits=self.search_config.folds, shuffle=True)

        for train_index, validation_index in fold.split(images):
            train_fold_images, validation_fold_images = images[train_index], images[validation_index]
            train_fold_labels, validation_fold_labels = labels[train_index], labels[validation_index]

            train_fold = (tf.data.Dataset.from_tensor_slices((train_fold_images, train_fold_labels))
                          .batch(self.search_config.batch_size)
                          .prefetch(buffer_size=tf.data.AUTOTUNE))

            validation_fold = (tf.data.Dataset.from_tensor_slices((validation_fold_images, validation_fold_labels))
                               .batch(self.search_config.batch_size)
                               .prefetch(buffer_size=tf.data.AUTOTUNE))

            tuner.search(
                train_fold,
                epochs=self.search_config.epochs,
                validation_data=validation_fold,
                batch_size=self.search_config.batch_size,
                verbose=self.search_config.log_level,
                callbacks=self.search_config.callbacks,
            )

        best_hyperparams = tuner.get_best_hyperparameters(num_trials=1)[0]
        model_instance = model.build(best_hyperparams)

        history = model_instance.fit(
            self.train_data,
            epochs=self.final_fit_config.epochs,
            batch_size=self.final_fit_config.batch_size,
            verbose=self.final_fit_config.log_level,
            validation_data=self.validation_data,
        )

        return model_instance, history


class ImageAugmentation:
    def __init__(self,
                 flip=True,
                 rotation=0.1,
                 zoom=0.1,
                 contrast=0.1,
                 translation=0.1,
                 brightness=0.2,
                 noise=0.1):
        self.data_augmentation = keras.Sequential()

        if flip:
            self.data_augmentation.add(keras.layers.RandomFlip("horizontal"))

        if rotation > 0:
            self.data_augmentation.add(keras.layers.RandomRotation(rotation))

        if zoom > 0:
            self.data_augmentation.add(keras.layers.RandomZoom(zoom))

        if contrast > 0:
            self.data_augmentation.add(keras.layers.RandomContrast(contrast))

        if translation > 0:
            self.data_augmentation.add(keras.layers.RandomTranslation(translation, translation))

        if brightness > 0:
            self.data_augmentation.add(keras.layers.RandomBrightness(brightness))

        if noise > 0:
            self.data_augmentation.add(keras.layers.GaussianNoise(noise))

    def apply(self, dataset, training=True):
        return dataset.map(lambda x, y: (self.data_augmentation(x, training=training), y))

class ImageRescaler:
    def __init__(self, scale=1./255):
        self.rescale_layer = keras.layers.Rescaling(scale)

    def apply(self, dataset):
        return dataset.map(lambda x, y: (self.rescale_layer(x), y))