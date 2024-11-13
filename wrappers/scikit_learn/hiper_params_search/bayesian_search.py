import time

from skopt import BayesSearchCV

from wrappers.scikit_learn import CommonHiperParamsSearcher, Searcher


class BayesianHipperParamsSearcher(CommonHiperParamsSearcher):
    """
    Implementação wrapper da busca BayesSearchCV o qual é detalhado na `documentação do scikit-optimize <https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html>`_.
    """

    def __init__(self, number_iterations: int, n_jobs: int = -1, log_level: int = 0):
        """
        :param number_iterations: Número máximo de iterações realizadas para testar as combinações de valores dos parâmetros.
        """
        super().__init__(n_jobs, log_level)
        self.number_iterations = number_iterations

    def search_hiper_parameters(self, estimator, params, data_x, data_y, cv, scoring: str) -> Searcher:
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