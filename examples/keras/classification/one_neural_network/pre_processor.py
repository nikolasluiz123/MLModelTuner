import kagglehub
import keras

from examples.keras.classification.one_neural_network.testing_one_neural_network import img_height, img_width, batch_size
from wrappers.keras.pre_processing.pre_processor import KerasDataPreProcessor


class ExamplePreProcessor(KerasDataPreProcessor):

    def _on_execute(self) -> tuple:
        path_image = kagglehub.dataset_download('hasyimabdillah/workoutexercises-images')

        train = keras.utils.image_dataset_from_directory(
            path_image,
            validation_split=0.2,
            subset="training",
            image_size=(img_height, img_width),
            batch_size=batch_size
        )

        validation = keras.utils.image_dataset_from_directory(
            path_image,
            validation_split=0.2,
            subset="validation",
            image_size=(img_height, img_width),
            batch_size=batch_size
        )

        return train, validation