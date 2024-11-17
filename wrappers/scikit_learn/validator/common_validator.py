from abc import abstractmethod
from typing import Generic

from wrappers.common.history_manager.common_history_manager import CommonValResult
from wrappers.common.validator.common_validator import CommonValidator


class ScikitLearnCommonValidator(CommonValidator, Generic[CommonValResult]):
    """
    Classe base para validadores de modelos do scikit-learn.
    """

    def __init__(self, log_level: int = 1, n_jobs: int = -1):
        """
        :param n_jobs: Número de thread que serão utilizadas para realizar a validação
        """
        super().__init__(log_level)
        self.n_jobs = n_jobs

    @abstractmethod
    def validate(self,
                 searcher,
                 data_x,
                 data_y,
                 cv=None,
                 scoring=None) -> CommonValResult:
        """
        Função que realiza a validação do modelo. Deve retornar um objeto contendo os dados com o resultado da validação.

        :param searcher: Objeto retornado pelo processo de busca de hiperparâmetros.
        :param data_x: Conjunto de dados de entrada (features) para validação.
        :param data_y: Conjunto de dados de saída (rótulos) para validação.
        :param cv: Estratégia de validação cruzada a ser utilizada (opcional).
        :param scoring: Métrica de avaliação a ser utilizada (opcional).
        """
