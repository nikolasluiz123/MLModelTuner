import kagglehub
import keras
import numpy as np
from tensorflow.python.keras.mixed_precision.policy import set_global_policy

set_global_policy('mixed_float16')
seed = 42
img_height = 128
img_width = 128
batch_size = 128

keras.utils.set_random_seed(seed)

path_image = kagglehub.dataset_download('hasyimabdillah/workoutexercises-images')

dados = keras.utils.image_dataset_from_directory(
    path_image,
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

classes_names = dados.class_names

teste = keras.utils.image_dataset_from_directory(
    'final_validation_images',
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

model = keras.models.load_model('modelo_final_v2.keras')

predictions = model.predict(teste)
predicted_classes = np.argmax(predictions, axis=1)

predicted_class_names = [classes_names[i] for i in predicted_classes]

true_labels = []
for _, label in teste:
    true_labels.extend(label.numpy())

true_class_names = [classes_names[i] for i in true_labels]

# Exibir os resultados
print("Predicted class names:", predicted_class_names)
print("True class names:", true_class_names)

# # Calcular matriz de confusão
# conf_matrix = confusion_matrix(true_labels, predicted_classes)
#
# # Plotar a matriz de confusão como um heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix - Final Model Validation")
# plt.show()