from abc import ABC


class CommonValidator(ABC):

    def __init__(self, log_level: int = 0):
        self.log_level = log_level

        self.start_validation_best_model_time = 0
        self.end_validation_best_model_time = 0