import kagglehub
import keras
import tensorflow as tf
from keras.src.callbacks import EarlyStopping

from keras_tuner import Hyperband

from examples.keras.common_classes import CrossValidator
from examples.keras.graficos import plotar_resultados
from examples.keras.test_variation_1_images.classes import TestVariation1, ImageAugmentation, ImageRescaler

########################################################################################################################
#                                   Definição dos Valores Fixos de Processamento                                       #
########################################################################################################################

seed = 42

img_height = 128
img_width = 128
input_shape = (img_height, img_width, 3)

batch_size = 32

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

treino = treino.prefetch(buffer_size=tf.data.AUTOTUNE)
validacao = validacao.prefetch(buffer_size=tf.data.AUTOTUNE)

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

treino = augmentation.apply(treino, training=True)
validacao = augmentation.apply(validacao, training=False)

########################################################################################################################
#                                           Reescalando as Imagens                                                     #
########################################################################################################################
rescaler = ImageRescaler(scale=1./255)

treino = rescaler.apply(treino)
validacao = rescaler.apply(validacao)

########################################################################################################################
#                                      Criando o Modelo Final para Avaliar                                             #
########################################################################################################################
modelo = TestVariation1(base_model=modelo_base, num_classes=22)

########################################################################################################################
#                                          Buscando os Melhores Parâmetros                                             #
########################################################################################################################
early_stopping = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)

tuner = Hyperband(
    modelo,
    objective='val_accuracy',
    max_epochs=5,
    factor=3,
    directory='search_hp_history',
    project_name='test_variation_1_images'
)

tuner.search(treino,
             epochs=10,
             validation_data=validacao,
             batch_size=batch_size,
             verbose=1,
             callbacks=[early_stopping])

########################################################################################################################
#                                       Validação Cruzada do Melhor Modelo                                             #
########################################################################################################################

cross_validator = CrossValidator(
    model_class=TestVariation1,
    base_model=modelo_base,
    num_classes=22,
    num_folds=5,
    batch_size=batch_size,
    epochs=10,
    tuner=tuner
)

resultados = cross_validator.run(treino)

plotar_resultados(resultados['history'], 'resultados_media_folds')