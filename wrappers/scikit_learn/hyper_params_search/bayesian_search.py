import time

from skopt import BayesSearchCV

from wrappers.scikit_learn.hyper_params_search.common_hyper_params_searcher import ScikitLearnCommonHyperParamsSearcher, \
    ScikitLearnSearcher


class ScikitLearnBayesianHyperParamsSearcher(ScikitLearnCommonHyperParamsSearcher):
    """
    Implementação wrapper da busca BayesSearchCV.
    """

    def __init__(self, number_iterations: int, n_jobs: int = -1, log_level: int = 0):
        """
        :param number_iterations: Número máximo de iterações realizadas para testar as combinações de valores dos parâmetros.
        """
        super().__init__(n_jobs, log_level)
        self.number_iterations = number_iterations

    def search_hyper_parameters(self, estimator, params, data_x, data_y, cv, scoring: str) -> ScikitLearnSearcher:
        self.start_search_parameter_time = time.time()

        search = BayesSearchCV(estimator=estimator,
                               search_spaces=params,
                               cv=cv,
                               n_jobs=self.n_jobs,
                               verbose=self.log_level,
                               scoring=scoring,
                               n_iter=self.number_iterations)

        search.fit(X=data_x, y=data_y)

        self.end_search_parameter_time = time.time()

        return search