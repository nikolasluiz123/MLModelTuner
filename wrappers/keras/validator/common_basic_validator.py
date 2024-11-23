from abc import ABC, abstractmethod
from typing import Generic

from wrappers.common.validator.common_validator import CommonValidator
from wrappers.keras.history_manager.common_history_manager import KerasValResult


class KerasCommonBasicValidator(CommonValidator, Generic[KerasValResult], ABC):
    """
    Implementação base para implementação de validadores de redes neurais específicos de acordo com a necessidade,
    por exemplo, classificação e regressão.
    """

    def __init__(self,
                 epochs: int,
                 batch_size: int,
                 callbacks: list,
                 log_level: int = 0):
        super().__init__(log_level)
        self.epochs = epochs
        self.batch_size = batch_size
        self.callbacks = callbacks

    @abstractmethod
    def validate(self, model_instance, train_data, validation_data) -> KerasValResult:
        """
        Função que deve realizar a validação do modelo executando um fit com dados de treino e validação, dessa forma é
        possível obter métricas interessantes com dois conjuntos de dados.

        Após a realização do treino e validação o objeto de resultado é montado com esses dados e retornado.
        """
