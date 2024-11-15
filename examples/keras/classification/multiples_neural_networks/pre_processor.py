
import kagglehub
import keras

from wrappers.keras.pre_processing.pre_processor import KerasDataPreProcessor

img_height = 128
img_width = 128
input_shape = (img_height, img_width, 3)

seed = 42

batch_size = 256

class ExamplePreProcessor(KerasDataPreProcessor):

    def _on_execute_train_process(self) -> tuple:
        path_image = kagglehub.dataset_download('hasyimabdillah/workoutexercises-images')

        train = keras.utils.image_dataset_from_directory(
            path_image,
            validation_split=0.2,
            subset="training",
            image_size=(img_height, img_width),
            batch_size=batch_size,
            seed=seed
        )

        validation = keras.utils.image_dataset_from_directory(
            path_image,
            validation_split=0.2,
            subset="validation",
            image_size=(img_height, img_width),
            batch_size=batch_size,
            seed=seed
        )

        return train, validation