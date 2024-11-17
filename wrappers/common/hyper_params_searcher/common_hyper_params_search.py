from abc import ABC


class CommonHyperParamsSearch(ABC):

    def __init__(self, log_level: int = 0):
        self.log_level = log_level

        self.start_search_parameter_time = 0
        self.end_search_parameter_time = 0
