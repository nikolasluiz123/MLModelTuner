import time

import numpy as np
from sklearn.model_selection import cross_val_score

from wrappers.scikit_learn.validator.common_validator import BaseValidator, Result
from wrappers.scikit_learn.validator.results.cross_validation import ScikitLearnCrossValidationResult


class CrossValidator(BaseValidator):
    """
    Validador que utiliza validação cruzada para avaliar modelos.

    Esta classe herda de `BaseValidator` e implementa a função de validação utilizando
    a validação cruzada. O validador calcula métricas de desempenho como média, desvio padrão,
    mediana, variância, erro padrão e os valores mínimo e máximo dos scores obtidos durante
    a validação.

    :param log_level: Nível de log para controle de saída de informações (padrão é 0).
    :param n_jobs: Número de trabalhos a serem executados em paralelo. -1 significa usar todos os processadores.
    """

    def __init__(self,
                 log_level: int = 0,
                 n_jobs: int = -1):
        """
        Inicializa um novo validador de validação cruzada.

        :param log_level: Nível de log para controle de saída de informações.
        :param n_jobs: Número de trabalhos a serem executados em paralelo.
        """
        super().__init__(log_level, n_jobs)

    def validate(self,
                 searcher,
                 data_x,
                 data_y,
                 cv=None,
                 scoring=None) -> Result | None:
        """
        Valida o modelo utilizando validação cruzada.

        Esta função executa a validação cruzada usando o buscador fornecido e retorna um
        objeto `CrossValidationResult` contendo as métricas de avaliação.

        :param searcher: O objeto que contém o modelo a ser avaliado.
        :param data_x: Conjunto de dados de entrada (features) para validação.
        :param data_y: Conjunto de dados de saída (rótulos) para validação.
        :param cv: Estratégia de validação cruzada a ser utilizada.
        :param scoring: Métrica de avaliação a ser utilizada.

        :raises Exception: Levanta uma exceção se `cv` ou `scoring` forem None.

        :return: Um objeto `CrossValidationResult` contendo as métricas de validação.
        """
        if cv is None:
            raise Exception("The parameter cv can't be None")

        if scoring is None:
            raise Exception("The parameter scoring can't be None")

        self.start_best_model_validation = time.time()

        scores = cross_val_score(estimator=searcher,
                                 X=data_x,
                                 y=data_y,
                                 cv=cv,
                                 n_jobs=self.n_jobs,
                                 verbose=self.log_level,
                                 scoring=scoring)

        self.end_best_model_validation = time.time()

        result = ScikitLearnCrossValidationResult(
            mean=np.mean(scores),
            standard_deviation=np.std(scores),
            median=np.median(scores),
            variance=np.var(scores),
            standard_error=np.std(scores) / np.sqrt(len(scores)),
            min_max_score=(round(float(np.min(scores)), 4), round(float(np.max(scores)), 4)),
            estimator=searcher.best_estimator_,
            scoring=scoring
        )

        return result
