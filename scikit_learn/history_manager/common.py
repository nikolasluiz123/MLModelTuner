import json
import os
import pickle
from abc import abstractmethod, ABC
from typing import Generic

from scikit_learn.validator.common_validator import Result


class HistoryManager(ABC, Generic[Result]):

    def __init__(self, output_directory: str, models_directory: str, params_file_name: str):
        self.output_directory = output_directory
        self.models_directory = os.path.join(self.output_directory, models_directory)
        self.params_file_name = params_file_name

    @abstractmethod
    def save_result(self,
                    classifier_result: Result,
                    feature_selection_time: str,
                    search_time: str,
                    validation_time: str,
                    scoring: str,
                    features: list[str]):
        ...

    def _create_output_dir(self):
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        if not os.path.exists(self.models_directory):
            os.makedirs(self.models_directory)

    def _save_dictionary_in_json(self, dictionary):
        output_path = os.path.join(self.output_directory, f"{self.params_file_name}.json")

        if os.path.exists(output_path):
            with open(output_path, 'r') as file:
                data = json.load(file)
        else:
            data = []

        data.append(dictionary)

        with open(output_path, 'w') as file:
            json.dump(data, file, indent=4)

    def has_history(self) -> bool:
        output_path = os.path.join(self.output_directory, f"{self.params_file_name}.json")

        if not os.path.exists(output_path):
            return False

        with open(output_path, 'r') as file:
            data = json.load(file)
            return len(data) > 0

    @abstractmethod
    def load_validation_result_from_history(self, index: int = -1) -> Result:
        ...

    def get_dictionary_from_json(self, index):
        output_path = os.path.join(self.output_directory, f"{self.params_file_name}.json")

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"O arquivo {self.params_file_name}.json não foi encontrado no diretório {self.output_directory}.")

        with open(output_path, 'r') as file:
            data = json.load(file)

        if index < -1 or index >= len(data):
            raise IndexError(f"Índice {index} fora dos limites. O arquivo contém {len(data)} entradas.")

        result_dict = data[index]

        return result_dict

    def _save_model(self, estimator):
        history_len = self._get_history_len()
        output_path = os.path.join(self.models_directory, f"model_{history_len}.pkl")

        with open(output_path, 'wb') as file:
            pickle.dump(estimator, file)

    def get_saved_model(self, version: int):
        output_path = os.path.join(self.models_directory, f"model_{version}.pkl")

        with open(output_path, 'rb') as f:
            return pickle.load(f)

    def _get_history_len(self) -> int:
        output_path = os.path.join(self.output_directory, f"{self.params_file_name}.json")

        if not os.path.exists(output_path):
            return 0

        with open(output_path, 'r') as file:
            data = json.load(file)
            return len(data)
