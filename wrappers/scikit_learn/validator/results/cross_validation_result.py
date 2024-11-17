from typing import Any

from wrappers.common.validator.results.common_validation_result import CommonValidationResult


class ScikitLearnCrossValidationResult(CommonValidationResult):
    """
    Implementação do objeto de validação cruzada utilizada nos processos que se baseiam nas implementações do scikit-learn.
    """

    def __init__(self,
                 mean: float,
                 standard_deviation: float,
                 median: float,
                 variance: float,
                 standard_error: float,
                 min_max_score: tuple[float, float],
                 estimator,
                 scoring: str):
        """
        :param mean: Média dos resultados.
        :param standard_deviation: Desvio padrão dos resultados.
        :param median: Mediana dos resultados.
        :param variance: Variância dos resultados.
        :param standard_error: Erro padrão dos resultados.
        :param min_max_score: Valor mínimo e máximo dos resultados.
        :param estimator: Modelo avaliado.
        :param scoring: Critério de avaiação, por exemplo, accuracy (classificação) ou neg_mean_squared_error (regressão)
        """

        self.mean = mean
        self.standard_deviation = standard_deviation
        self.median = median
        self.variance = variance
        self.standard_error = standard_error
        self.min_max_score = min_max_score
        self.estimator = estimator
        self.scoring = scoring

    def append_data(self, pipeline_infos: dict[str, Any]) -> dict[str, Any]:
        pipeline_infos['scoring'] = self.scoring
        pipeline_infos['mean'] = self.mean
        pipeline_infos['standard_deviation'] = self.standard_deviation
        pipeline_infos['median'] = self.median
        pipeline_infos['variance'] = self.variance
        pipeline_infos['standard_error'] = self.standard_error
        pipeline_infos['min_max_score'] = self.min_max_score
        pipeline_infos['scoring'] = self.scoring

        return pipeline_infos
