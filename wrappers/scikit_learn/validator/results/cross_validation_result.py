from typing import Any

from wrappers.common.validator.results.common_validation_result import CommonValidationResult


class ScikitLearnCrossValidationResult(CommonValidationResult):

    def __init__(self,
                 mean: float,
                 standard_deviation: float,
                 median: float,
                 variance: float,
                 standard_error: float,
                 min_max_score: tuple[float, float],
                 estimator,
                 scoring: str):
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
