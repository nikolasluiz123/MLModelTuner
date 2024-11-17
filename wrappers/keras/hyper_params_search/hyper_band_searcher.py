from keras.src.callbacks import Callback
from keras_tuner import HyperModel, Hyperband

from wrappers.keras.hyper_params_search.common_hyper_params_searcher import KerasCommonHyperParamsSearcher


class KerasHyperBandSearcher(KerasCommonHyperParamsSearcher):

    def __init__(self,
                 objective: str | list[str],
                 directory: str,
                 project_name: str,
                 epochs: int,
                 batch_size: int,
                 callbacks: list[Callback],
                 factor: int,
                 max_epochs: int,
                 hyper_band_iterations: int,
                 log_level: int = 0):
        super().__init__(objective, directory, project_name, epochs, batch_size, callbacks, log_level)
        self.factor = factor
        self.max_epochs = max_epochs
        self.hyper_band_iterations = hyper_band_iterations

    def _on_execute(self, train_data, validation_data, model: HyperModel):
        tuner = Hyperband(
            model,
            objective=self.objective,
            factor=self.factor,
            directory=self.directory,
            project_name=self.project_name,
            max_epochs=self.max_epochs,
            hyperband_iterations=self.hyper_band_iterations
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
        return [
            'hyperband_iterations',
            'max_epochs',
            'min_epochs',
            'factor'
        ]
