import keras

from keras_tuner import HyperModel
from keras.src.layers import Flatten, Dense, BatchNormalization, Dropout


class ExampleKerasHyperModelV1(HyperModel):

    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

    def build(self, hp):
        self.base_model.trainable = False
        last_layer = self.base_model.get_layer('mixed5')

        model_extension = Flatten()(last_layer.output)

        dense_units_1 = hp.Int('dense_units_1', min_value=32, max_value=512, step=32)
        dense_units_2 = hp.Int('dense_units_2', min_value=32, max_value=512, step=32)
        dropout_rate1 = hp.Float('dropout_rate1', min_value=0.1, max_value=0.5, step=0.1)
        dropout_rate2 = hp.Float('dropout_rate2', min_value=0.1, max_value=0.5, step=0.1)

        model_extension = Dense(units=dense_units_1, activation='relu')(model_extension)
        model_extension = BatchNormalization(name='batch_norm_1')(model_extension)
        model_extension = Dropout(rate=dropout_rate1)(model_extension)
        model_extension = Dense(units=dense_units_2, activation='relu')(model_extension)
        model_extension = BatchNormalization(name='batch_norm_2')(model_extension)
        model_extension = Dropout(rate=dropout_rate2)(model_extension)
        model_extension = Dense(self.num_classes, activation='softmax')(model_extension)

        model = keras.models.Model(inputs=self.base_model.input, outputs=model_extension)

        learning_rate = hp.Float(name='learning_rate', min_value=0.0001, max_value=0.01)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        return model

class ExampleKerasHyperModelV2(HyperModel):

    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

    def build(self, hp):
        self.base_model.trainable = False
        last_layer = self.base_model.get_layer('mixed5')

        model_extension = Flatten()(last_layer.output)

        dense_units_1 = hp.Int('dense_units_1', min_value=16, max_value=256, step=32)
        dense_units_2 = hp.Int('dense_units_2', min_value=16, max_value=256, step=32)
        dropout_rate1 = hp.Float('dropout_rate1', min_value=0.1, max_value=0.8, step=0.1)
        dropout_rate2 = hp.Float('dropout_rate2', min_value=0.1, max_value=0.8, step=0.1)

        model_extension = Dense(units=dense_units_1, activation='relu')(model_extension)
        model_extension = BatchNormalization(name='batch_norm_1')(model_extension)
        model_extension = Dropout(rate=dropout_rate1)(model_extension)
        model_extension = Dense(units=dense_units_2, activation='relu')(model_extension)
        model_extension = BatchNormalization(name='batch_norm_2')(model_extension)
        model_extension = Dropout(rate=dropout_rate2)(model_extension)
        model_extension = Dense(self.num_classes, activation='softmax')(model_extension)

        model = keras.models.Model(inputs=self.base_model.input, outputs=model_extension)

        learning_rate = hp.Float(name='learning_rate', min_value=0.0001, max_value=0.01)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        return model

class ExampleKerasHyperModelV3(HyperModel):

    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

    def build(self, hp):
        self.base_model.trainable = False
        last_layer = self.base_model.get_layer('mixed5')

        model_extension = Flatten()(last_layer.output)

        dense_units_1 = hp.Int('dense_units_1', min_value=16, max_value=256, step=32)
        dropout_rate1 = hp.Float('dropout_rate1', min_value=0.1, max_value=0.8, step=0.1)

        model_extension = Dense(units=dense_units_1, activation='relu')(model_extension)
        model_extension = BatchNormalization(name='batch_norm_1')(model_extension)
        model_extension = Dropout(rate=dropout_rate1)(model_extension)
        model_extension = Dense(self.num_classes, activation='softmax')(model_extension)

        model = keras.models.Model(inputs=self.base_model.input, outputs=model_extension)

        learning_rate = hp.Float(name='learning_rate', min_value=0.0001, max_value=0.01)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        return model

class ExampleKerasHyperModelV4(HyperModel):

    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

    def build(self, hp):
        self.base_model.trainable = False
        last_layer = self.base_model.get_layer('mixed5')

        model_extension = Flatten()(last_layer.output)

        dense_units_1 = hp.Int('dense_units_1', min_value=16, max_value=1024, step=64)
        dropout_rate1 = hp.Float('dropout_rate1', min_value=0.1, max_value=0.8, step=0.1)

        model_extension = Dense(units=dense_units_1, activation='relu')(model_extension)
        model_extension = BatchNormalization(name='batch_norm_1')(model_extension)
        model_extension = Dropout(rate=dropout_rate1)(model_extension)
        model_extension = Dense(self.num_classes, activation='softmax')(model_extension)

        model = keras.models.Model(inputs=self.base_model.input, outputs=model_extension)

        learning_rate = hp.Float(name='learning_rate', min_value=0.0001, max_value=0.01)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        return model

class ExampleKerasHyperModelV5(HyperModel):

    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

    def build(self, hp):
        self.base_model.trainable = False
        last_layer = self.base_model.get_layer('mixed1')

        model_extension = Flatten()(last_layer.output)

        dense_units_1 = hp.Int('dense_units_1', min_value=16, max_value=1024, step=64)
        dropout_rate1 = hp.Float('dropout_rate1', min_value=0.1, max_value=0.8, step=0.1)

        model_extension = Dense(units=dense_units_1, activation='relu')(model_extension)
        model_extension = BatchNormalization(name='batch_norm_1')(model_extension)
        model_extension = Dropout(rate=dropout_rate1)(model_extension)
        model_extension = Dense(self.num_classes, activation='softmax')(model_extension)

        model = keras.models.Model(inputs=self.base_model.input, outputs=model_extension)

        learning_rate = hp.Float(name='learning_rate', min_value=0.0001, max_value=0.01)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        return model
