from keras.src.callbacks import Callback
from keras_tuner import HyperModel, RandomSearch

from wrappers.keras.hyper_params_search.common_hyper_params_searcher import KerasCommonHyperParamsSearcher


class KerasRandomSearcher(KerasCommonHyperParamsSearcher):
    """
    Implementação wrapper da busca de hiperparâmetros utilizando RandomSearch do Keras Tuner.
    """

    def __init__(self,
                 objective: str | list[str],
                 directory: str,
                 project_name: str,
                 epochs: int,
                 batch_size: int,
                 callbacks: list,
                 max_trials: int,
                 log_level: int = 0):
        """
        :param max_trials: Número máximo de tentativas realizadas para tentar obter o modelo com os melhores parâmetros
        """

        super().__init__(objective, directory, project_name, epochs, batch_size, callbacks, log_level)
        self.max_trials = max_trials

    def _on_execute(self, train_data, validation_data, model: HyperModel):
        tuner = RandomSearch(
            model,
            objective=self.objective,
            directory=self.directory,
            project_name=self.project_name,
            max_trials=self.max_trials,
        )

        tuner.search(
            train_data,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.log_level,
            callbacks=self.callbacks
        )

        best_hyperparams = tuner.get_best_hyperparameters(num_trials=1)[0]
        model_instance = model.build(best_hyperparams)

        return model_instance

    def get_fields_oracle_json_file(self) -> list[str]:
        return []