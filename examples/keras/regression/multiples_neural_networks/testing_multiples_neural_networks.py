from keras.src.callbacks import EarlyStopping

from examples.keras.regression.multiples_neural_networks.model_class import WeightSuggestorLSTMV1, \
    WeightSuggestorLSTMV5, WeightSuggestorLSTMV4, WeightSuggestorLSTMV3, WeightSuggestorLSTMV2
from examples.keras.regression.multiples_neural_networks.pre_processor import ExampleDataPreProcessor
from wrappers.keras.history_manager.regressor_history_manager import KerasRegressorHistoryManager
from wrappers.keras.hyper_params_search.hyper_band_searcher import KerasHyperBandSearcher
from wrappers.keras.process_manager.pipeline import KerasPipeline
from wrappers.keras.process_manager.regressor_multi_process_manager import KerasRegressorMultProcessManager
from wrappers.keras.validator.additional_validator import KerasAdditionalRegressorValidator
from wrappers.keras.validator.basic_regressor_validator import KerasBasicRegressorValidator

########################################################################################################################
#                                               Preparando Implementações                                              #
########################################################################################################################

pre_processor = ExampleDataPreProcessor('../data/workout_train_data.csv')
early_stopping_validation = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

validator = KerasBasicRegressorValidator(
    epochs=200,
    batch_size=ExampleDataPreProcessor.BATCH_SIZE,
    log_level=1,
    callbacks=[early_stopping_validation]
)

params_searcher_1 = KerasHyperBandSearcher(
    objective='val_loss',
    directory='search_params_1',
    project_name='model_example_1',
    epochs=10,
    batch_size=ExampleDataPreProcessor.BATCH_SIZE,
    log_level=1,
    callbacks=[],
    max_epochs=20,
    factor=3,
    hyper_band_iterations=1
)

params_searcher_2 = KerasHyperBandSearcher(
    objective='val_loss',
    directory='search_params_2',
    project_name='model_example_2',
    epochs=10,
    batch_size=ExampleDataPreProcessor.BATCH_SIZE,
    log_level=1,
    callbacks=[],
    max_epochs=20,
    factor=3,
    hyper_band_iterations=1
)

params_searcher_3 = KerasHyperBandSearcher(
    objective='val_loss',
    directory='search_params_3',
    project_name='model_example_3',
    epochs=10,
    batch_size=ExampleDataPreProcessor.BATCH_SIZE,
    log_level=1,
    callbacks=[],
    max_epochs=20,
    factor=3,
    hyper_band_iterations=1
)

params_searcher_4 = KerasHyperBandSearcher(
    objective='val_loss',
    directory='search_params_4',
    project_name='model_example_4',
    epochs=10,
    batch_size=ExampleDataPreProcessor.BATCH_SIZE,
    log_level=1,
    callbacks=[],
    max_epochs=20,
    factor=3,
    hyper_band_iterations=1
)

params_searcher_5 = KerasHyperBandSearcher(
    objective='val_loss',
    directory='search_params_5',
    project_name='model_example_5',
    epochs=10,
    batch_size=ExampleDataPreProcessor.BATCH_SIZE,
    log_level=1,
    callbacks=[],
    max_epochs=20,
    factor=3,
    hyper_band_iterations=1
)

history_manager_model_example_1 = KerasRegressorHistoryManager(output_directory='history_model_example_1',
                                                               models_directory='models',
                                                               best_params_file_name='best_executions')

history_manager_model_example_2 = KerasRegressorHistoryManager(output_directory='history_model_example_2',
                                                               models_directory='models',
                                                               best_params_file_name='best_executions')

history_manager_model_example_3 = KerasRegressorHistoryManager(output_directory='history_model_example_3',
                                                               models_directory='models',
                                                               best_params_file_name='best_executions')

history_manager_model_example_4 = KerasRegressorHistoryManager(output_directory='history_model_example_4',
                                                               models_directory='models',
                                                               best_params_file_name='best_executions')

history_manager_model_example_5 = KerasRegressorHistoryManager(output_directory='history_model_example_5',
                                                               models_directory='models',
                                                               best_params_file_name='best_executions')

########################################################################################################################
#                                               Preparando Pipelines                                                   #
########################################################################################################################

pipelines = [
    KerasPipeline(
        model=WeightSuggestorLSTMV1(),
        data_pre_processor=pre_processor,
        params_searcher=params_searcher_1,
        validator=validator,
        history_manager=history_manager_model_example_1
    ),
    KerasPipeline(
        model=WeightSuggestorLSTMV2(),
        data_pre_processor=pre_processor,
        params_searcher=params_searcher_2,
        validator=validator,
        history_manager=history_manager_model_example_2
    ),
    KerasPipeline(
        model=WeightSuggestorLSTMV3(),
        data_pre_processor=pre_processor,
        validator=validator,
        params_searcher=params_searcher_3,
        history_manager=history_manager_model_example_3
    ),
    KerasPipeline(
        model=WeightSuggestorLSTMV4(),
        data_pre_processor=pre_processor,
        validator=validator,
        params_searcher=params_searcher_4,
        history_manager=history_manager_model_example_4
    ),
    KerasPipeline(
        model=WeightSuggestorLSTMV5(),
        data_pre_processor=pre_processor,
        validator=validator,
        params_searcher=params_searcher_5,
        history_manager=history_manager_model_example_5
    )
]

history_manager_best_model = KerasRegressorHistoryManager(output_directory='best_executions',
                                                          models_directory='best_models',
                                                          best_params_file_name='best_executions')
manager = KerasRegressorMultProcessManager(
    pipelines=pipelines,
    seed=ExampleDataPreProcessor.SEED,
    history_manager=history_manager_best_model,
    history_index=-1,
    save_history=False,
    delete_trials_after_execution=True
)

manager.process_pipelines()

########################################################################################################################
#                                               Validação Adicional                                                    #
########################################################################################################################

model = history_manager_best_model.get_saved_model(history_manager_best_model.get_history_len())
additional_validation_data = pre_processor.get_data_as_numpy(pre_processor.get_data_additional_validation())
additional_validator = KerasAdditionalRegressorValidator(model_instance=model,
                                                         data=additional_validation_data,
                                                         prefix_file_names='final_model',
                                                         validation_results_directory='additional_validations',
                                                         scaler=pre_processor.scaler_y)
additional_validator.validate()
