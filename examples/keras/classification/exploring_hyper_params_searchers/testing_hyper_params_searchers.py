import keras
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau

from examples.keras.classification.exploring_hyper_params_searchers.model_class import ExampleKerasHyperModel
from examples.keras.classification.exploring_hyper_params_searchers.pre_processor import input_shape, \
    ExamplePreProcessor, batch_size, seed
from wrappers.keras.history_manager.classifier_history_manager import KerasClassifierHistoryManager
from wrappers.keras.hyper_params_search.grid_searcher import KerasGridSearcher
from wrappers.keras.hyper_params_search.hyper_band_searcher import KerasHyperBandSearcher
from wrappers.keras.hyper_params_search.random_searcher import KerasRandomSearcher
from wrappers.keras.process_manager.classifier_mult_process_manager import KerasClassifierMultProcessManager
from wrappers.keras.process_manager.pipeline import KerasPipeline
from wrappers.keras.validator.basic_classifier_validator import KerasBasicClassifierValidator
from wrappers.keras.validator.classifier_validator import KerasAdditionalClassifierValidator

########################################################################################################################
#                                           Definições Estáticas do Teste                                              #
########################################################################################################################

num_classes = 22

base_model = keras.applications.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=input_shape
)

########################################################################################################################
#                                       Definição do Pipeline de Execução                                              #
########################################################################################################################

early_stopping = EarlyStopping(monitor='val_loss', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

pre_processor = ExamplePreProcessor()
example_keras_hyper_model = ExampleKerasHyperModel(base_model=base_model, num_classes=num_classes)

keras_hyper_band_searcher = KerasHyperBandSearcher(objective='val_loss',
                                                   directory='search_params',
                                                   project_name='hyper_band_search',
                                                   epochs=10,
                                                   batch_size=batch_size,
                                                   log_level=1,
                                                   callbacks=[early_stopping, reduce_lr],
                                                   max_epochs=10,
                                                   factor=3,
                                                   hyper_band_iterations=1)

keras_random_searcher = KerasRandomSearcher(objective='val_loss',
                                            directory='search_params',
                                            project_name='random_search',
                                            epochs=10,
                                            batch_size=batch_size,
                                            log_level=1,
                                            callbacks=[early_stopping, reduce_lr],
                                            max_trials=30)

keras_grid_searcher = KerasGridSearcher(objective='val_loss',
                                        directory='search_params',
                                        project_name='grid_search',
                                        epochs=10,
                                        batch_size=batch_size,
                                        log_level=1,
                                        callbacks=[early_stopping, reduce_lr],
                                        max_trials=30)

validator = KerasBasicClassifierValidator(epochs=50,
                                          batch_size=batch_size,
                                          log_level=1,
                                          callbacks=[early_stopping])

history_manager_hyper_band_searcher = KerasClassifierHistoryManager(output_directory='history_hyper_band_searcher',
                                                                    models_directory='models',
                                                                    best_executions_file_name='best_executions')

history_manager_random_searcher = KerasClassifierHistoryManager(output_directory='history_random_searcher',
                                                                models_directory='models',
                                                                best_executions_file_name='best_executions')

history_manager_grid_searcher = KerasClassifierHistoryManager(output_directory='history_grid_searcher',
                                                              models_directory='models',
                                                              best_executions_file_name='best_executions')

pipelines = [
    KerasPipeline(
        model=example_keras_hyper_model,
        data_pre_processor=pre_processor,
        params_searcher=keras_hyper_band_searcher,
        validator=validator,
        history_manager=history_manager_hyper_band_searcher
    ),
    KerasPipeline(
        model=example_keras_hyper_model,
        data_pre_processor=pre_processor,
        params_searcher=keras_random_searcher,
        validator=validator,
        history_manager=history_manager_random_searcher
    ),
    KerasPipeline(
        model=example_keras_hyper_model,
        data_pre_processor=pre_processor,
        params_searcher=keras_grid_searcher,
        validator=validator,
        history_manager=history_manager_grid_searcher
    )
]

########################################################################################################################
#                                   Preparando Manager para Realizar a Execução                                        #
########################################################################################################################

history_manager_best_model = KerasClassifierHistoryManager(output_directory='best_executions',
                                                           models_directory='best_models',
                                                           best_executions_file_name='best_executions')
manager = KerasClassifierMultProcessManager(
    pipelines=pipelines,
    seed=seed,
    history_manager=history_manager_best_model,
    history_index=None,
    save_history=True,
    delete_trials_after_execution=True
)

manager.process_pipelines()

########################################################################################################################
#                                   Validação Adicional para Avaliar o Modelo                                          #
########################################################################################################################

result = history_manager_best_model.get_validation_result(-1)
final_model = history_manager_best_model.get_saved_model(history_manager_best_model.get_history_len())

data = pre_processor.get_data_additional_validation()
additional_validator = KerasAdditionalClassifierValidator(model_instance=final_model,
                                                          model=example_keras_hyper_model,
                                                          history_dict=result.history,
                                                          data=data)
additional_validator.validate()
