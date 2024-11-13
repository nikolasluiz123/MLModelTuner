import keras
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau

from examples.keras.classification.one_neural_network.model_class import ExampleKerasHyperModel
from examples.keras.classification.one_neural_network.pre_processor import ExamplePreProcessor
from wrappers.keras.config.configurators import HyperBandConfig, SearchConfig, FinalFitConfig
from wrappers.keras.history_manager.common import KerasHistoryManager
from wrappers.keras.process_manager.classifier_mult_process_manager import KerasClassifierMultProcessManager
from wrappers.keras.process_manager.pipeline import KerasPipeline
from wrappers.keras.validator.classifier_cross_validator import ClassifierKerasCrossValidator

img_height = 128
img_width = 128
input_shape = (img_height, img_width, 3)

batch_size = 128

num_classes = 22

base_model = keras.applications.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=input_shape
)

pre_processor = ExamplePreProcessor()
model = ExampleKerasHyperModel(base_model=base_model, num_classes=num_classes)
validator = ClassifierKerasCrossValidator()

early_stopping = EarlyStopping(monitor='val_loss', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

pipeline = KerasPipeline(
    model=model,
    data_pre_processor=pre_processor,
    validator=validator,
    hyper_band_config=HyperBandConfig(
        objective='val_loss',
        factor=3,
        max_epochs=10,
        directory='executions',
        project_name='test_model_example_1'
    ),
    search_config=SearchConfig(
        epochs=5,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        folds=5,
        log_level=1,
        stratified=True
    ),
    final_fit_config=FinalFitConfig(
        epochs=15,
        batch_size=batch_size * 3,
        log_level=1
    ),
    history_manager=KerasHistoryManager(
        output_directory='executions_model_example_1',
        models_directory='best_models_example_1',
        best_executions_file_name='best_executions_example_1'
    )
)

manager = KerasClassifierMultProcessManager(
    pipelines=pipeline,
    seed=42,
    history_manager=KerasHistoryManager(
        output_directory='best_executions',
        models_directory='best_models',
        best_executions_file_name='best_executions'
    )
)

manager.process_pipelines()
