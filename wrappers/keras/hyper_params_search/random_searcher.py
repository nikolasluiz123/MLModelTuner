from keras.src.callbacks import Callback
from keras_tuner import HyperModel, RandomSearch

from wrappers.keras.hyper_params_search.common_searcher import KerasCommonHyperParamsSearcher


class KerasRandomSearcher(KerasCommonHyperParamsSearcher):

    def __init__(self,
                 objective: str | list[str],
                 directory: str,
                 project_name: str,
                 epochs: int,
                 batch_size: int,
                 log_level: int,
                 callbacks: list[Callback],
                 max_trials: int):
        super().__init__(objective, directory, project_name, epochs, batch_size, log_level, callbacks)
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