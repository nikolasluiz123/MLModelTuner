from typing import final

import keras
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau

from examples.keras.classification.one_neural_network.model_class import ExampleKerasHyperModel
from examples.keras.classification.one_neural_network.pre_processor import ExamplePreProcessor, input_shape, batch_size, \
    seed
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
#                                       Definição do Pipeline de Execução                                              #
########################################################################################################################

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
        max_epochs=5,
        directory='executions',
        project_name='model_example_1'
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
        epochs=5,
        batch_size=128,
        log_level=1
    ),
    history_manager=KerasClassifierHistoryManager(
        output_directory='history_model_1',
        models_directory='models',
        best_executions_file_name='best_executions'
    )
)

########################################################################################################################
#                                   Preparando Manager para Realizar a Execução                                        #
########################################################################################################################

history_manager_best_model = KerasClassifierHistoryManager(output_directory='best_executions',
                                                           models_directory='best_models',
                                                           best_executions_file_name='best_executions')
manager = KerasClassifierMultProcessManager(
    pipelines=pipeline,
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
