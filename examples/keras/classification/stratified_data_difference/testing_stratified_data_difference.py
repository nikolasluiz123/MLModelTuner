import keras
import tensorflow as tf
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau

from examples.keras.classification.stratified_data_difference.model_class import ExampleKerasHyperModel
from examples.keras.classification.stratified_data_difference.pre_processor import ExamplePreProcessor, input_shape, \
    batch_size, seed
from wrappers.keras.config.configurators import HyperBandConfig, SearchConfig, FinalFitConfig
from wrappers.keras.history_manager.classifier_history_manager import KerasClassifierHistoryManager
from wrappers.keras.process_manager.classifier_mult_process_manager import KerasClassifierMultProcessManager
from wrappers.keras.process_manager.pipeline import KerasPipeline
from wrappers.keras.validator.classifier_cross_validator import ClassifierKerasCrossValidator, \
    KerasAdditionalClassifierValidator

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
#                                       Definição dos Pipelines de Execução                                            #
########################################################################################################################

pre_processor = ExamplePreProcessor()
model = ExampleKerasHyperModel(base_model=base_model, num_classes=num_classes)
validator = ClassifierKerasCrossValidator()

early_stopping_search = EarlyStopping(monitor='val_loss', restore_best_weights=True)
reduce_lr_search = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
early_stopping_final_fit = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

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

search_config_model_1 = SearchConfig(epochs=5,
                                     batch_size=batch_size,
                                     callbacks=[early_stopping_search, reduce_lr_search],
                                     folds=5,
                                     log_level=1,
                                     stratified=True)

search_config_model_2 = SearchConfig(epochs=5,
                                     batch_size=batch_size,
                                     callbacks=[early_stopping_search, reduce_lr_search],
                                     folds=5,
                                     log_level=1,
                                     stratified=False)

final_fit_config = FinalFitConfig(epochs=15, batch_size=batch_size, log_level=1, callbacks=[early_stopping_final_fit])

history_manager_model_example_1 = KerasClassifierHistoryManager(output_directory='executions_model_example_1',
                                                                models_directory='best_models_example_1',
                                                                best_executions_file_name='best_executions_example_1')

history_manager_model_example_2 = KerasClassifierHistoryManager(output_directory='executions_model_example_2',
                                                                models_directory='best_models_example_2',
                                                                best_executions_file_name='best_executions_example_2')

pipelines = [
    KerasPipeline(
        model=model,
        data_pre_processor=pre_processor,
        validator=validator,
        hyper_band_config=hyper_band_config_1,
        search_config=search_config_model_1,
        final_fit_config=final_fit_config,
        history_manager=history_manager_model_example_1
    ),
    KerasPipeline(
        model=model,
        data_pre_processor=pre_processor,
        validator=validator,
        hyper_band_config=hyper_band_config_2,
        search_config=search_config_model_2,
        final_fit_config=final_fit_config,
        history_manager=history_manager_model_example_2
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
    history_manager=history_manager_best_model
)

manager.process_pipelines()

########################################################################################################################
#                                   Validação Adicional para Avaliar o Modelo                                          #
########################################################################################################################

result = history_manager_best_model.get_validation_result(-1)
final_model = history_manager_best_model.get_saved_model(history_manager_best_model.get_history_len())

data = pre_processor.get_data_additional_validation()
additional_validator = KerasAdditionalClassifierValidator(model_instance=final_model,
                                                          model=model,
                                                          history_dict=result.history,
                                                          data=data)
additional_validator.validate()
