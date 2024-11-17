from abc import ABC, abstractmethod
from typing import Any


class CommonValidationResult(ABC):
    """
    Implementação padrão para representar o resultado de uma validação realizada com alguma implementação específica
    de CommonValidator.
    """

    @abstractmethod
    def append_data(self, pipeline_infos: dict[str, Any]) -> dict[str, Any]:
        """
        Função que deve ser implementada obrigatoriamente por todas as classes de resultado dessa validação para que
        seja possível adicionar os dados desse objeto em um dicionário que é exibido em alguns momentos da execução dos
        pipelines em forma de DataFrame.
        """