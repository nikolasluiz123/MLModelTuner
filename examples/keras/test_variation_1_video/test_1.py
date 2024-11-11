import kagglehub
import keras
import tensorflow as tf
from keras.src.callbacks import EarlyStopping
from keras_tuner import Hyperband

from examples.keras.test_variation_1_video.classes import TestVariation1Video
from examples.keras.test_variation_1_video.funcoes import extract_frames_from_videos_in_directory, load_video_dataset

########################################################################################################################
#                                   Definição dos Valores Fixos de Processamento                                       #
########################################################################################################################

seed = 42
img_height = 128
img_width = 128
sequence_length = 10
batch_size = 32

keras.utils.set_random_seed(seed)

########################################################################################################################
#                                          Baixando os Dados do Kaggle                                                 #
########################################################################################################################

path_video = kagglehub.dataset_download('hasyimabdillah/workoutfitness-video')

extract_frames_from_videos_in_directory(path_video, 'frames', frame_rate=12)

########################################################################################################################
#                                               Definição dos Dados                                                    #
########################################################################################################################

treino = load_video_dataset(
    directory='frames',
    sequence_length=sequence_length,
    img_height=img_height,
    img_width=img_width,
    batch_size=batch_size)

treino = treino.prefetch(buffer_size=tf.data.AUTOTUNE)

########################################################################################################################
#                                      Criando o Modelo Final para Avaliar                                             #
########################################################################################################################

modelo = TestVariation1Video(num_classes=22,
                             sequence_length=sequence_length,
                             frame_height=img_height,
                             frame_width=img_width)

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
    project_name='test_variation_1_video'
)

tuner.search(treino,
             epochs=10,
             # validation_data=validacao,
             batch_size=batch_size,
             verbose=1,
             callbacks=[early_stopping])
