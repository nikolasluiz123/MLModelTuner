from keras import Input, Model

from keras.src.layers import Dense, GlobalAveragePooling2D, \
    TimeDistributed, LSTM, Dropout, RandomFlip, RandomRotation, RandomZoom, RandomContrast, Conv2D, MaxPooling2D
from keras_tuner import HyperModel


class TestVariation1Video(HyperModel):

    def __init__(self, num_classes, sequence_length, frame_height, frame_width):
        super().__init__()
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.frame_height = frame_height
        self.frame_width = frame_width

    def build(self, hp):
        input_layer = Input(shape=(self.sequence_length, self.frame_height, self.frame_width, 3))

        augmented = TimeDistributed(RandomFlip('horizontal'))(input_layer)
        augmented = TimeDistributed(RandomRotation(0.2))(augmented)
        augmented = TimeDistributed(RandomZoom(0.2))(augmented)
        augmented = TimeDistributed(RandomContrast(0.2))(augmented)

        conv2d_filters_1 = hp.Int(name='conv2d_filters_1', min_value=32, max_value=256)
        conv2d_filters_2 = hp.Int(name='conv2d_filters_2', min_value=32, max_value=256)
        conv2d_filters_3 = hp.Int(name='conv2d_filters_3', min_value=32, max_value=256)

        lstm_units_1 = hp.Int(name='lstm_units_1', min_value=32, max_value=256)
        dense_units_1 = hp.Int(name='dense_units_1', min_value=32, max_value=256)
        dropout_rate_1 = hp.Float(name='dropout_rate_1', min_value=0.1, max_value=0.5)

        x = TimeDistributed(Conv2D(conv2d_filters_1, (3, 3), activation='relu'))(augmented)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
        x = TimeDistributed(Conv2D(conv2d_filters_2, (3, 3), activation='relu'))(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
        x = TimeDistributed(Conv2D(conv2d_filters_3, (3, 3), activation='relu'))(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
        x = TimeDistributed(GlobalAveragePooling2D())(x)

        x = LSTM(units=lstm_units_1)(x)
        x = Dense(units=dense_units_1, activation='relu')(x)
        x = Dropout(rate=dropout_rate_1)(x)

        output_layer = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model