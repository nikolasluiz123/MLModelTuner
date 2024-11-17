from abc import ABC, abstractmethod
from typing import Any


class CommonPipeline(ABC):
    """
    Implementação base de pipelines de execução, contendo os comportamentos comuns obrigatórios entre os pipelines de
    cada biblioteca.
    """

    @abstractmethod
    def get_dictionary_pipeline_data(self) -> dict[str, Any]:
        """
        Função que deve ser implementada pelo pipeline para que seja possível recuperar os dados armazenados nele em forma
        de dicionário.

        Essa informação é utilizada normalmente para exibição no console durante a execução do pipeline e também ao
        fim de todas as N execuções. Não necessariamente todos os dados do pipeline precisam estar contidos nesse dicionário,
        adicione nele apenas o que for relevante e/ou que possa diferenciar os pipelines.
        """

    @abstractmethod
    def get_execution_times(self) -> tuple:
        """
        Função que deve ser implementada para que seja possível obter os tempos de execução de cada processo específico
        realizado ao executar o pipeline.

        O retorno é esperado em formato de tupla, ou seja, no momento de executar o retorno os dados de cada tempo de
        execução devem estar separados por vírgula.
        """