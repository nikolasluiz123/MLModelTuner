from abc import ABC


class CommonValidator(ABC):
    """
    Implementação que contém tudo que é comum nos validadores dos modelos das diferentes bibliotecas.
    """

    def __init__(self, log_level: int = 0):
        """
        :param log_level: Inteiro que representa o quanto de informação será exibida no console durante a execução da
                          validação do modelo.

        Atributos Internos:
            start_validation_best_model_time: Armazena o tempo de início da validação, em segundos.
            end_validation_best_model_time: Armazena o tempo de término da validação, em segundos.
        """
        self.log_level = log_level

        self.start_validation_best_model_time = 0
        self.end_validation_best_model_time = 0