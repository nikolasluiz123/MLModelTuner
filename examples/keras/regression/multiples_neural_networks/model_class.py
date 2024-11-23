import keras
from keras import Input
from keras.src.layers import LSTM, Dense
from keras.src.optimizers import Adam
from keras_tuner import HyperModel

from examples.keras.regression.multiples_neural_networks.pre_processor import ExampleDataPreProcessor


class WeightSuggestorLSTMV1(HyperModel):

    def build(self, hp):
        lstm_units_1 = hp.Int(name='lstm_units_1', min_value=32, max_value=1024, step=32)
        dropout_lstm_1 = hp.Float(name='dropout_lstm_1', min_value=0.1, max_value=0.5, step=0.1)

        dense_units_1 = hp.Int(name='lstm_units_1', min_value=32, max_value=512, step=32)
        dropout_dense_1 = hp.Float(name='dropout_lstm_1', min_value=0.1, max_value=0.5, step=0.1)

        learning_rate = hp.Float(name='learning_rate', min_value=0.0001, max_value=0.01, step=0.0001)

        model = keras.models.Sequential()

        model.add(Input(shape=(ExampleDataPreProcessor.SEQUENCE_LENGTH,
                               ExampleDataPreProcessor.FEATURES_NUMBER)))

        model.add(LSTM(units=lstm_units_1, activation='tanh'))
        model.add(keras.layers.Dropout(dropout_lstm_1))

        model.add(Dense(units=dense_units_1, activation='relu'))
        model.add(keras.layers.Dropout(dropout_dense_1))

        model.add(Dense(units=1))

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.MeanAbsoluteError()]
        )

        return model


class WeightSuggestorLSTMV2(HyperModel):

    def build(self, hp):
        lstm_units_1 = hp.Int(name='lstm_units_1', min_value=32, max_value=1024, step=32)
        dropout_lstm_1 = hp.Float(name='dropout_lstm_1', min_value=0.1, max_value=0.5, step=0.1)

        lstm_units_2 = hp.Int(name='lstm_units_2', min_value=32, max_value=512, step=32)
        dropout_lstm_2 = hp.Float(name='dropout_lstm_2', min_value=0.1, max_value=0.5, step=0.1)

        dense_units_1 = hp.Int(name='lstm_units_1', min_value=32, max_value=512, step=32)
        dropout_dense_1 = hp.Float(name='dropout_lstm_1', min_value=0.1, max_value=0.5, step=0.1)

        learning_rate = hp.Float(name='learning_rate', min_value=0.0001, max_value=0.01, step=0.0001)

        model = keras.models.Sequential()

        model.add(Input(shape=(ExampleDataPreProcessor.SEQUENCE_LENGTH,
                               ExampleDataPreProcessor.FEATURES_NUMBER)))

        model.add(LSTM(units=lstm_units_1, activation='tanh', return_sequences=True))
        model.add(keras.layers.Dropout(dropout_lstm_1))

        model.add(LSTM(units=lstm_units_2, activation='tanh'))
        model.add(keras.layers.Dropout(dropout_lstm_2))

        model.add(Dense(units=dense_units_1, activation='relu'))
        model.add(keras.layers.Dropout(dropout_dense_1))

        model.add(Dense(units=1))

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.MeanAbsoluteError()]
        )

        return model


class WeightSuggestorLSTMV3(HyperModel):

    def build(self, hp):
        lstm_units_1 = hp.Int(name='lstm_units_1', min_value=32, max_value=1024, step=32)
        dropout_lstm_1 = hp.Float(name='dropout_lstm_1', min_value=0.1, max_value=0.5, step=0.1)

        lstm_units_2 = hp.Int(name='lstm_units_2', min_value=32, max_value=512, step=32)
        dropout_lstm_2 = hp.Float(name='dropout_lstm_2', min_value=0.1, max_value=0.5, step=0.1)

        dense_units_1 = hp.Int(name='lstm_units_1', min_value=32, max_value=512, step=32)
        dropout_dense_1 = hp.Float(name='dropout_lstm_1', min_value=0.1, max_value=0.5, step=0.1)

        dense_units_2 = hp.Int(name='lstm_units_2', min_value=32, max_value=256, step=32)
        dropout_dense_2 = hp.Float(name='dropout_lstm_2', min_value=0.1, max_value=0.5, step=0.1)

        learning_rate = hp.Float(name='learning_rate', min_value=0.0001, max_value=0.01, step=0.0001)

        model = keras.models.Sequential()

        model.add(Input(shape=(ExampleDataPreProcessor.SEQUENCE_LENGTH,
                               ExampleDataPreProcessor.FEATURES_NUMBER)))

        model.add(LSTM(units=lstm_units_1, activation='tanh', return_sequences=True))
        model.add(keras.layers.Dropout(dropout_lstm_1))

        model.add(LSTM(units=lstm_units_2, activation='tanh'))
        model.add(keras.layers.Dropout(dropout_lstm_2))

        model.add(Dense(units=dense_units_1, activation='relu'))
        model.add(keras.layers.Dropout(dropout_dense_1))

        model.add(Dense(units=dense_units_2, activation='relu'))
        model.add(keras.layers.Dropout(dropout_dense_2))

        model.add(Dense(units=1))

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.MeanAbsoluteError()]
        )

        return model


class WeightSuggestorLSTMV4(HyperModel):

    def build(self, hp):
        lstm_units_1 = hp.Int(name='lstm_units_1', min_value=32, max_value=1024, step=32)
        dropout_lstm_1 = hp.Float(name='dropout_lstm_1', min_value=0.1, max_value=0.5, step=0.1)

        lstm_units_2 = hp.Int(name='lstm_units_2', min_value=32, max_value=512, step=32)
        dropout_lstm_2 = hp.Float(name='dropout_lstm_2', min_value=0.1, max_value=0.5, step=0.1)

        lstm_units_3 = hp.Int(name='lstm_units_3', min_value=32, max_value=256, step=32)
        dropout_lstm_3 = hp.Float(name='dropout_lstm_3', min_value=0.1, max_value=0.5, step=0.1)

        dense_units_1 = hp.Int(name='lstm_units_1', min_value=32, max_value=512, step=32)
        dropout_dense_1 = hp.Float(name='dropout_lstm_1', min_value=0.1, max_value=0.5, step=0.1)

        dense_units_2 = hp.Int(name='lstm_units_2', min_value=32, max_value=256, step=32)
        dropout_dense_2 = hp.Float(name='dropout_lstm_2', min_value=0.1, max_value=0.5, step=0.1)

        learning_rate = hp.Float(name='learning_rate', min_value=0.0001, max_value=0.01, step=0.0001)

        model = keras.models.Sequential()

        model.add(Input(shape=(ExampleDataPreProcessor.SEQUENCE_LENGTH,
                               ExampleDataPreProcessor.FEATURES_NUMBER)))

        model.add(LSTM(units=lstm_units_1, activation='tanh', return_sequences=True))
        model.add(keras.layers.Dropout(dropout_lstm_1))

        model.add(LSTM(units=lstm_units_2, activation='tanh', return_sequences=True))
        model.add(keras.layers.Dropout(dropout_lstm_2))

        model.add(LSTM(units=lstm_units_3, activation='tanh'))
        model.add(keras.layers.Dropout(dropout_lstm_3))

        model.add(Dense(units=dense_units_1, activation='relu'))
        model.add(keras.layers.Dropout(dropout_dense_1))

        model.add(Dense(units=dense_units_2, activation='relu'))
        model.add(keras.layers.Dropout(dropout_dense_2))

        model.add(Dense(units=1))

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.MeanAbsoluteError()]
        )

        return model


class WeightSuggestorLSTMV5(HyperModel):

    def build(self, hp):
        lstm_units_1 = hp.Int(name='lstm_units_1', min_value=32, max_value=1024, step=32)
        dropout_lstm_1 = hp.Float(name='dropout_lstm_1', min_value=0.1, max_value=0.5, step=0.1)

        lstm_units_2 = hp.Int(name='lstm_units_2', min_value=32, max_value=512, step=32)
        dropout_lstm_2 = hp.Float(name='dropout_lstm_2', min_value=0.1, max_value=0.5, step=0.1)

        lstm_units_3 = hp.Int(name='lstm_units_3', min_value=32, max_value=256, step=32)
        dropout_lstm_3 = hp.Float(name='dropout_lstm_3', min_value=0.1, max_value=0.5, step=0.1)

        dense_units_1 = hp.Int(name='lstm_units_1', min_value=32, max_value=512, step=32)
        dropout_dense_1 = hp.Float(name='dropout_lstm_1', min_value=0.1, max_value=0.5, step=0.1)

        dense_units_2 = hp.Int(name='lstm_units_3', min_value=32, max_value=256, step=32)
        dropout_dense_2 = hp.Float(name='dropout_lstm_3', min_value=0.1, max_value=0.5, step=0.1)

        dense_units_3 = hp.Int(name='lstm_units_3', min_value=32, max_value=128, step=32)
        dropout_dense_3 = hp.Float(name='dropout_lstm_3', min_value=0.1, max_value=0.5, step=0.1)

        learning_rate = hp.Float(name='learning_rate', min_value=0.0001, max_value=0.01, step=0.0001)

        model = keras.models.Sequential()

        model.add(Input(shape=(ExampleDataPreProcessor.SEQUENCE_LENGTH,
                               ExampleDataPreProcessor.FEATURES_NUMBER)))

        model.add(LSTM(units=lstm_units_1, activation='tanh', return_sequences=True))
        model.add(keras.layers.Dropout(dropout_lstm_1))

        model.add(LSTM(units=lstm_units_2, activation='tanh', return_sequences=True))
        model.add(keras.layers.Dropout(dropout_lstm_2))

        model.add(LSTM(units=lstm_units_3, activation='tanh'))
        model.add(keras.layers.Dropout(dropout_lstm_3))

        model.add(Dense(units=dense_units_1, activation='relu'))
        model.add(keras.layers.Dropout(dropout_dense_1))

        model.add(Dense(units=dense_units_2, activation='relu'))
        model.add(keras.layers.Dropout(dropout_dense_2))

        model.add(Dense(units=dense_units_3, activation='relu'))
        model.add(keras.layers.Dropout(dropout_dense_3))

        model.add(Dense(units=1))

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.MeanAbsoluteError()]
        )

        return model