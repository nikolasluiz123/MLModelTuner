import tensorflow as tf
import keras
import numpy as np
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import KFold

class CrossValidator:
    def __init__(self, model_class, base_model, num_classes, num_folds=5, batch_size=32, epochs=10, tuner=None):
        self.model_class = model_class
        self.base_model = base_model
        self.num_classes = num_classes
        self.num_folds = num_folds
        self.batch_size = batch_size
        self.epochs = epochs
        self.tuner = tuner
        self.best_hp = None

    def train_and_evaluate(self, model, train_data, val_data):
        # Treinando o modelo
        history = model.fit(train_data,
                            validation_data=val_data,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            verbose=1)

        # Avaliando o modelo
        val_loss, val_accuracy = model.evaluate(val_data, verbose=0)

        return history, val_accuracy

    def run(self, treino, model_path="final_model.keras"):
        if self.tuner:
            self.best_hp = self.tuner.oracle.get_best_trials()[0].hyperparameters
        else:
            self.best_hp = None

        accuracies = []
        all_history = {
            'accuracy': np.zeros(self.epochs),
            'val_accuracy': np.zeros(self.epochs),
            'loss': np.zeros(self.epochs),
            'val_loss': np.zeros(self.epochs)
        }

        # kf.split(treino) gera índices, não precisamos converter para lista ou numpy
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)

        x, y = self.__dataset_to_array(treino)

        for fold, (train_index, val_index) in enumerate(kf.split(x, y), 1):
            print(f"Training fold {fold}")

            # Divida os dados em treino e validação para o fold atual
            train_fold = tf.data.Dataset.from_tensor_slices((x[train_index], y[train_index])).batch(self.batch_size)
            val_fold = tf.data.Dataset.from_tensor_slices((x[val_index], y[val_index])).batch(self.batch_size)

            # Criando o modelo
            model = self.model_class(base_model=self.base_model, num_classes=self.num_classes)
            model_instance = model.build(self.best_hp) if self.best_hp else model.build(None)

            # Treinando e avaliando o modelo
            history, accuracy = self.train_and_evaluate(model_instance, train_fold, val_fold)
            accuracies.append(accuracy)

            # Somando os resultados de cada fold para depois calcular a média
            all_history['accuracy'] += np.array(history.history['accuracy'])
            all_history['val_accuracy'] += np.array(history.history['val_accuracy'])
            all_history['loss'] += np.array(history.history['loss'])
            all_history['val_loss'] += np.array(history.history['val_loss'])

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        # Calculando a média dos históricos de todos os folds
        for key in all_history:
            all_history[key] /= self.num_folds

        # Treinando o modelo final com os melhores hiperparâmetros em todos os dados
        print("Treinando o modelo final com todos os dados de treino...")
        final_model = self.model_class(base_model=self.base_model, num_classes=self.num_classes)
        final_model_instance = final_model.build(self.best_hp) if self.best_hp else final_model.build(None)
        final_model_instance.fit(treino, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

        # Salvando o modelo final
        final_model_instance.save(model_path)
        print(f"Modelo final salvo em {model_path}")

        return {
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "history": all_history,
        }

    @staticmethod
    def __dataset_to_array(dataset):
        images, labels = [], []

        for batch_images, batch_labels in dataset:
            images.extend(batch_images.numpy())
            labels.extend(batch_labels.numpy())

        return np.array(images), np.array(labels)