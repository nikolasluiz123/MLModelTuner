import keras
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau

from examples.keras.classification.one_neural_network.model_class import ExampleKerasHyperModel
from examples.keras.classification.one_neural_network.pre_processor import ExamplePreProcessor, input_shape, batch_size, \
    seed
from wrappers.keras.history_manager.classifier_history_manager import KerasClassifierHistoryManager
from wrappers.keras.hyper_params_search.hyper_band_searcher import KerasHyperBandSearcher
from wrappers.keras.process_manager.classifier_multi_process_manager import KerasClassifierMultProcessManager
from wrappers.keras.process_manager.pipeline import KerasPipeline
from wrappers.keras.validator.basic_classifier_validator import KerasBasicClassifierValidator
from wrappers.keras.validator.classifier_additional_validator import KerasAdditionalClassifierValidator

########################################################################################################################
#                                           Definições Estáticas do Teste                                              #
########################################################################################################################

num_classes = 22

########################################################################################################################
#                                           Modelo Pré Treinado Utilizado                                              #
########################################################################################################################

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

searcher = KerasHyperBandSearcher(objective='val_loss',
                                  directory='search_params',
                                  project_name='model_example_1',
                                  epochs=10,
                                  batch_size=batch_size,
                                  log_level=1,
                                  callbacks=[early_stopping, reduce_lr],
                                  max_epochs=10,
                                  factor=3,
                                  hyper_band_iterations=1)

validator = KerasBasicClassifierValidator(epochs=50,
                                          batch_size=batch_size,
                                          log_level=1,
                                          callbacks=[early_stopping])

model_1_history_manager = KerasClassifierHistoryManager(output_directory='history_model_1',
                                                        models_directory='models',
                                                        best_params_file_name='best_executions')
pipeline = KerasPipeline(
    model=example_keras_hyper_model,
    data_pre_processor=pre_processor,
    params_searcher=searcher,
    validator=validator,
    history_manager=model_1_history_manager
)

########################################################################################################################
#                                   Preparando Manager para Realizar a Execução                                        #
########################################################################################################################

history_manager_best_model = KerasClassifierHistoryManager(output_directory='best_executions',
                                                           models_directory='best_models',
                                                           best_params_file_name='best_executions')
manager = KerasClassifierMultProcessManager(
    pipelines=pipeline,
    seed=seed,
    history_manager=history_manager_best_model,
    history_index=-1,
    save_history=True,
    delete_trials_after_execution=True
)

manager.process_pipelines()

########################################################################################################################
#                                   Validação Adicional para Avaliar o Modelo                                          #
########################################################################################################################

result = history_manager_best_model.load_validation_result_from_history(-1)
final_model = history_manager_best_model.get_saved_model(history_manager_best_model.get_history_len())

additional_validator = KerasAdditionalClassifierValidator(model_instance=final_model,
                                                          data=pre_processor.get_data_additional_validation(),
                                                          prefix_file_names='final_model',
                                                          validation_results_directory='additional_validations')
additional_validator.validate()
