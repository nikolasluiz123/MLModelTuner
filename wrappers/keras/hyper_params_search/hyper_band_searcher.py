from keras.src.callbacks import Callback
from keras_tuner import HyperModel, Hyperband

from wrappers.keras.hyper_params_search.common_hyper_params_searcher import KerasCommonHyperParamsSearcher


class KerasHyperBandSearcher(KerasCommonHyperParamsSearcher):
    """
    Implementação wrapper da busca de hiperparâmetros utilizando Hyperband do Keras Tuner.
    """

    def __init__(self,
                 objective: str | list[str],
                 directory: str,
                 project_name: str,
                 epochs: int,
                 batch_size: int,
                 callbacks: list,
                 factor: int,
                 max_epochs: int,
                 hyper_band_iterations: int,
                 log_level: int = 0):
        """
        :param factor: Define o fator pelo qual os recursos são reduzidos entre as rodadas dentro de cada execução.
                       Este parâmetro controla a agressividade da redução, valores menores resultam em maior número de
                       modelos avaliados inicialmente, enquanto valores maiores reduzem rapidamente o número de modelos
                       testados.

        :param max_epochs: Especifica o número máximo de épocas que um modelo pode usar durante a execução de cada tentativa.
                           Este parâmetro define o limite superior do recurso que será alocado para os modelos mais
                           promissores no final do processo.

        :param hyper_band_iterations: Número de vezes que todas as tentativas serão executadas. Vamos supor que `factor`
                                      foi definido como 3 e `max_epochs` foi definido como 10, serão executadas 30 tentativas,
                                      o número definido nesse parâmetro é quantas vezes serão executadas essas 30 tentativas.
        """
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
