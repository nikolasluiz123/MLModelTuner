import numpy as np

from wrappers.common.data_pre_processor.common_data_pre_processor import CommonDataPreProcessor
from wrappers.common.validator.common_additional_validator import CommonClassifierAdditionalValidator


class KerasAdditionalClassifierValidator(CommonClassifierAdditionalValidator):

    def __init__(self,
                 model_instance,
                 data_pre_processor: CommonDataPreProcessor,
                 show_graphics: bool = True):
        super().__init__(data_pre_processor, show_graphics)
        self.model_instance = model_instance

    def validate(self, show_graphic: bool = False):
        dataset = self.data_pre_processor.get_data_additional_validation()
        true_labels = []

        for _, label in dataset:
            true_labels.extend(label.numpy())

        predictions = self.model_instance.predict(dataset)
        predicted_classes = np.argmax(predictions, axis=1)

        classes_names = sorted(set(dataset.class_names))

        predicted_class_names = [classes_names[i] for i in predicted_classes]
        true_class_names = [classes_names[i] for i in true_labels]

        self._show_classification_report(predicted_class_names, true_class_names)
        self._show_confusion_matrix(predicted_class_names, true_class_names, classes_names)
