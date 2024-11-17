import time

from sklearn.model_selection import RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

from wrappers.scikit_learn.hiper_params_search.common_hyper_params_searcher import ScikitLearnCommonHyperParamsSearcher, \
    ScikitLearnSearcher


class ScikitLearnRandomCVHyperParamsSearcher(ScikitLearnCommonHyperParamsSearcher):
    """
    Implementação wrapper da busca RandomizedSearchCV.
    """

    def __init__(self, number_iterations: int, n_jobs: int = -1, log_level: int = 0):
        """
        :param number_iterations: Número máximo de iterações realizadas na busca do melhor modelo.
        """
        super().__init__(n_jobs, log_level)
        self.number_iterations = number_iterations

    def search_hyper_parameters(self, estimator, params, data_x, data_y, cv, scoring: str) -> ScikitLearnSearcher:
        self.start_search_parameter_time = time.time()

        search = RandomizedSearchCV(estimator=estimator,
                                    param_distributions=params,
                                    n_iter=self.number_iterations,
                                    cv=cv,
                                    n_jobs=self.n_jobs,
                                    verbose=self.log_level,
                                    scoring=scoring)

        search.fit(X=data_x, y=data_y)

        self.end_search_parameter_time = time.time()

        return search


class ScikitLearnHalvingRandomCVHyperParamsSearcher(ScikitLearnCommonHyperParamsSearcher):
    """
    Implementação wrapper da busca HalvingRandomSearchCV.
    """

    def __init__(self,
                 number_candidates,
                 resource='n_samples',
                 max_resources='auto',
                 min_resources='exhaust',
                 factor=3,
                 n_jobs: int = -1,
                 log_level: int = 0):
        """
        :param number_candidates: Número máximo de iterações (candidatos) realizadas na busca do melhor modelo. Também
                                  pode ser utilizado o valor `exhaust` e o processo será baseado em `max_resources` e
                                  `min_resources`.

        :param resource: Especifica o recurso a ser usado para avaliar o desempenho de cada combinação de hiperparâmetros.
                         Pode ser um número de amostras, tempo de treinamento ou qualquer outro recurso que se deseja limitar durante a
                         validação.

        :param max_resources: Define a quantidade máxima do recurso que pode ser utilizado durante a execução. Se você
                              especificar max_resources=100, a busca usará no máximo 100 amostras (ou outra métrica definida em resource)
                              durante a validação de cada combinação.

        :param min_resources: Determina a quantidade mínima do recurso que deve ser usada na primeira iteração da busca.
                              É o ponto de partida que garante que a busca comece com uma base sólida de amostras ou recursos.

        :param factor: Determina a proporção dos candidatos em cada iteração. Por padrão 1/3.
        """

        super().__init__(n_jobs, log_level)
        self.number_candidates = number_candidates
        self.resource = resource
        self.max_resources = max_resources
        self.min_resources = min_resources
        self.factor = factor

    def search_hyper_parameters(self, estimator, params, data_x, data_y, cv, scoring: str) -> ScikitLearnSearcher:
        self.start_search_parameter_time = time.time()

        search = HalvingRandomSearchCV(estimator=estimator,
                                       param_distributions=params,
                                       cv=cv,
                                       n_jobs=self.n_jobs,
                                       verbose=self.log_level,
                                       scoring=scoring,
                                       n_candidates=self.number_candidates,
                                       resource=self.resource,
                                       max_resources=self.max_resources,
                                       min_resources=self.min_resources,
                                       factor=self.factor)

        search.fit(X=data_x, y=data_y)

        self.end_search_parameter_time = time.time()

        return search
