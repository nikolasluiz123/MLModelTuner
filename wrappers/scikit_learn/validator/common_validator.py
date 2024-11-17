from abc import abstractmethod
from typing import Generic

from wrappers.common.history_manager.common_history_manager import CommonValResult
from wrappers.common.validator.common_validator import CommonValidator


class ScikitLearnCommonValidator(CommonValidator, Generic[CommonValResult]):
    """
    Classe base abstrata para validadores de modelos.

    Esta classe define a interface para os validadores de modelos de aprendizado de máquina,
    especificando métodos que devem ser implementados em subclasses. O validador pode
    incluir logs e configuração de paralelização.

    :param log_level: Nível de log para controle de saída de informações (padrão é 1).
    :param n_jobs: Número de trabalhos a serem executados em paralelo. -1 significa usar todos os processadores.
    """

    def __init__(self, log_level: int = 1, n_jobs: int = -1):
        """
        Inicializa um novo validador base.

        :param log_level: Nível de log para controle de saída de informações.
        :param n_jobs: Número de trabalhos a serem executados em paralelo.
        """
        super().__init__(log_level)
        self.n_jobs = n_jobs

    @abstractmethod
    def validate(self,
                 searcher,
                 data_x,
                 data_y,
                 cv=None,
                 scoring=None) -> CommonValResult | None:
        """
        Função abstrata para validar um modelo.

        As subclasses devem implementar esta função para realizar a validação do modelo
        utilizando o buscador fornecido.

        :param searcher: Objeto responsável pela busca de hiperparâmetros ou pela seleção de modelos.
        :param data_x: Conjunto de dados de entrada (features) para validação.
        :param data_y: Conjunto de dados de saída (rótulos) para validação.
        :param cv: Estratégia de validação cruzada a ser utilizada (opcional).
        :param scoring: Métrica de avaliação a ser utilizada (opcional).

        :return: Um objeto do tipo Result contendo os resultados da validação ou None se não for aplicável.
        """
        ...
