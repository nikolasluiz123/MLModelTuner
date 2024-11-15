from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau

from examples.keras.classification.multiples_neural_networks.model_class import *
from examples.keras.classification.multiples_neural_networks.pre_processor import ExamplePreProcessor, input_shape, \
    batch_size, seed
from wrappers.keras.config.configurators import HyperBandConfig, SearchConfig, FinalFitConfig
from wrappers.keras.history_manager.classifier_history_manager import KerasClassifierHistoryManager
from wrappers.keras.process_manager.classifier_mult_process_manager import KerasClassifierMultProcessManager
from wrappers.keras.process_manager.pipeline import KerasPipeline
from wrappers.keras.validator.classifier_cross_validator import ClassifierKerasCrossValidator

num_classes = 22

base_model = keras.applications.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=input_shape
)

pre_processor = ExamplePreProcessor()
validator = ClassifierKerasCrossValidator()

early_stopping_search = EarlyStopping(monitor='val_loss', restore_best_weights=True)
early_stopping_final_fit = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
reduce_lr_search = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

hyper_band_config_1 = HyperBandConfig(objective='val_loss',
                                      factor=3,
                                      max_epochs=10,
                                      directory='executions',
                                      project_name='test_model_example_1')

hyper_band_config_2 = HyperBandConfig(objective='val_loss',
                                      factor=3,
                                      max_epochs=10,
                                      directory='executions',
                                      project_name='test_model_example_2')

hyper_band_config_3 = HyperBandConfig(objective='val_loss',
                                      factor=3,
                                      max_epochs=10,
                                      directory='executions',
                                      project_name='test_model_example_3')

hyper_band_config_4 = HyperBandConfig(objective='val_loss',
                                      factor=3,
                                      max_epochs=10,
                                      directory='executions',
                                      project_name='test_model_example_4')

hyper_band_config_5 = HyperBandConfig(objective='val_loss',
                                      factor=3,
                                      max_epochs=10,
                                      directory='executions',
                                      project_name='test_model_example_5')

search_config_model = SearchConfig(epochs=5,
                                   batch_size=batch_size,
                                   callbacks=[early_stopping_search, reduce_lr_search],
                                   folds=5,
                                   log_level=1,
                                   stratified=True)

final_fit_config = FinalFitConfig(epochs=15,
                                  batch_size=batch_size,
                                  callbacks=[early_stopping_final_fit],
                                  log_level=1)

history_manager_model_example_1 = KerasClassifierHistoryManager(output_directory='executions_model_example_1',
                                                                models_directory='best_models_example_1',
                                                                best_executions_file_name='best_executions_example_1')

history_manager_model_example_2 = KerasClassifierHistoryManager(output_directory='executions_model_example_2',
                                                                models_directory='best_models_example_2',
                                                                best_executions_file_name='best_executions_example_2')

history_manager_model_example_3 = KerasClassifierHistoryManager(output_directory='executions_model_example_3',
                                                                models_directory='best_models_example_3',
                                                                best_executions_file_name='best_executions_example_3')

history_manager_model_example_4 = KerasClassifierHistoryManager(output_directory='executions_model_example_4',
                                                                models_directory='best_models_example_4',
                                                                best_executions_file_name='best_executions_example_4')

history_manager_model_example_5 = KerasClassifierHistoryManager(output_directory='executions_model_example_5',
                                                                models_directory='best_models_example_5',
                                                                best_executions_file_name='best_executions_example_5')

pipelines = [
    KerasPipeline(
        model=ExampleKerasHyperModelV1(base_model=base_model, num_classes=num_classes),
        data_pre_processor=pre_processor,
        validator=validator,
        hyper_band_config=hyper_band_config_1,
        search_config=search_config_model,
        final_fit_config=final_fit_config,
        history_manager=history_manager_model_example_1
    ),
    KerasPipeline(
        model=ExampleKerasHyperModelV2(base_model=base_model, num_classes=num_classes),
        data_pre_processor=pre_processor,
        validator=validator,
        hyper_band_config=hyper_band_config_2,
        search_config=search_config_model,
        final_fit_config=final_fit_config,
        history_manager=history_manager_model_example_2
    ),
    KerasPipeline(
        model=ExampleKerasHyperModelV3(base_model=base_model, num_classes=num_classes),
        data_pre_processor=pre_processor,
        validator=validator,
        hyper_band_config=hyper_band_config_3,
        search_config=search_config_model,
        final_fit_config=final_fit_config,
        history_manager=history_manager_model_example_3
    ),
    KerasPipeline(
        model=ExampleKerasHyperModelV4(base_model=base_model, num_classes=num_classes),
        data_pre_processor=pre_processor,
        validator=validator,
        hyper_band_config=hyper_band_config_4,
        search_config=search_config_model,
        final_fit_config=final_fit_config,
        history_manager=history_manager_model_example_4
    ),
    KerasPipeline(
        model=ExampleKerasHyperModelV5(base_model=base_model, num_classes=num_classes),
        data_pre_processor=pre_processor,
        validator=validator,
        hyper_band_config=hyper_band_config_5,
        search_config=search_config_model,
        final_fit_config=final_fit_config,
        history_manager=history_manager_model_example_5
    )
]

history_manager_best_model = KerasClassifierHistoryManager(output_directory='best_executions',
                                                           models_directory='best_models',
                                                           best_executions_file_name='best_executions')
manager = KerasClassifierMultProcessManager(
    pipelines=pipelines,
    seed=seed,
    history_manager=history_manager_best_model
)

manager.process_pipelines()
