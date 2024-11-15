from keras.src.applications.inception_v3 import InceptionV3
from keras.src.utils import plot_model

# Carregar o modelo InceptionV3
model = InceptionV3(weights='imagenet')

# Gerar uma imagem do modelo
plot_model(
    model,
    to_file="inceptionv3_summary.svg",  # Nome do arquivo de saída
    show_shapes=True,                  # Exibe os formatos dos tensores
    show_layer_names=True,             # Exibe os nomes das camadas
    dpi=96                             # Define a resolução da imagem
)