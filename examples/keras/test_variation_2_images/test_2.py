import kagglehub
import keras
import tensorflow as tf
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.mixed_precision.policy import set_global_policy

from examples.keras.common_classes import CrossValidator, HyperBandConfig, SearchConfig, FinalFitConfig, \
    ImageAugmentation, ImageRescaler
from examples.keras.graficos import plotar_resultados
from examples.keras.test_variation_2_images.classes import TestVariation2

set_global_policy('mixed_float16')

########################################################################################################################
#                                   Definição dos Valores Fixos de Processamento                                       #
########################################################################################################################

seed = 42

img_height = 128
img_width = 128
input_shape = (img_height, img_width, 3)

batch_size = 64

keras.utils.set_random_seed(seed)

########################################################################################################################
#                                          Baixando os Dados do Kaggle                                                 #
########################################################################################################################
path_image = kagglehub.dataset_download('hasyimabdillah/workoutexercises-images')

print("Imagens:", path_image)

########################################################################################################################
#                                               Definição dos Dados                                                    #
########################################################################################################################

treino = keras.utils.image_dataset_from_directory(
    path_image,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

validacao = keras.utils.image_dataset_from_directory(
    path_image,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

########################################################################################################################
#                                     Configurando o Modelo InceptionV3 como Base                                      #
########################################################################################################################
modelo_base = keras.applications.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=input_shape
)

modelo_base.trainable = False

########################################################################################################################
#                                     Aplicando Variações nas Imagens                                                  #
########################################################################################################################
augmentation = ImageAugmentation(rotation=0.1, zoom=0.1, contrast=0.1)

treino = treino.map(lambda x, y: (augmentation.data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)
validacao = validacao.map(lambda x, y: (augmentation.data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)

########################################################################################################################
#                                           Reescalando as Imagens                                                     #
########################################################################################################################
rescaler = ImageRescaler(scale=1./255)

treino = treino.map(lambda x, y: (rescaler.rescale_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
validacao = validacao.map(lambda x, y: (rescaler.rescale_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

########################################################################################################################
#                                      Criando o Modelo Final para Avaliar                                             #
########################################################################################################################
modelo = TestVariation2(base_model=modelo_base, num_classes=22)

########################################################################################################################
#                                          Buscando os Melhores Parâmetros                                             #
########################################################################################################################
early_stopping = EarlyStopping(monitor='val_loss', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

hyper_band_config = HyperBandConfig(
    objective='val_loss',
    factor=3,
    max_epochs=10,
    directory='../search_hp_history',
    project_name='test_variation_2_images'
)

search_config = SearchConfig(
    epochs=5,
    batch_size=batch_size,
    callbacks=[early_stopping, reduce_lr],
    folds=5,
    log_level=1
)

final_fit_config = FinalFitConfig(
    epochs=15,
    batch_size=batch_size * 4,
    log_level=1
)

validator = CrossValidator(
    train_data=treino,
    validation_data=validacao,
    hyper_band_config=hyper_band_config,
    search_config=search_config,
    final_fit_config=final_fit_config
)

model, history = validator.execute(modelo)
model.save('modelo_final_v2.keras')
plotar_resultados(history, 'resultado_v2')
