import json
import os
from abc import ABC
from typing import Generic, TypeVar

from wrappers.keras.validator.results.common import KerasValidationResult

Result = TypeVar('Result', bound=KerasValidationResult)

class KerasHistoryManager(ABC, Generic[Result]):

    def __init__(self,
                 output_directory: str,
                 models_directory: str,
                 best_executions_file_name: str):
        self.output_directory = output_directory
        self.models_directory = os.path.join(self.output_directory, models_directory)
        self.best_executions_file_name = best_executions_file_name

        self._create_output_dir()

    def _create_output_dir(self):
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        if not os.path.exists(self.models_directory):
            os.makedirs(self.models_directory)

    def save_result(self,
                    model,
                    final_fit_history,
                    hyper_band_executions_directory: str,
                    pre_processing_time: str,
                    validation_time: str):
        last_project = self.__get_last_project_from_hyper_band(hyper_band_executions_directory)
        oracle_data = self.__get_oracle_file(hyper_band_executions_directory, last_project)

        self.__save_best_model_execution_data(model, final_fit_history, oracle_data, pre_processing_time, validation_time)
        self.__save_keras_model(model)
        self.__delete_trials()

    def __save_best_model_execution_data(self,
                                         model,
                                         final_fit_history,
                                         oracle_data,
                                         pre_processing_time: str,
                                         validation_time: str):
        best_model_execution_data = {
            'model': type(model).__name__,
            'history': final_fit_history.history,
            'hyperband_iterations': oracle_data['hyperband_iterations'],
            'max_epochs': oracle_data['max_epochs'],
            'min_epochs': oracle_data['min_epochs'],
            'factor': oracle_data['factor'],
            'neural_structure': oracle_data['hyperparameters'],
            'pre_processing_time': pre_processing_time,
            'validation_time': validation_time,
        }

        output_path = os.path.join(self.output_directory, f"{self.best_executions_file_name}.json")

        if os.path.exists(output_path):
            with open(output_path, 'r') as file:
                data = json.load(file)
        else:
            data = []

        data.append(best_model_execution_data)

        with open(output_path, 'w') as file:
            json.dump(data, file, indent=4)

    def get_history_from_best_model_executions(self, index: int):
        executions = self.__get_dictionary_from_json(self.output_directory, index, self.best_executions_file_name)

        return executions['history']

    def __get_oracle_file(self, hyper_band_executions_directory, last_project):
        oracle_file_path = os.path.join(hyper_band_executions_directory, last_project, 'oracle.json')

        if os.path.isfile(oracle_file_path):
            with open(oracle_file_path, 'r') as file:
                oracle_data = json.load(file)
        else:
            print(f'O arquivo "oracle.json" não foi encontrado no diretório {hyper_band_executions_directory}.')

        return oracle_data

    def __get_dictionary_from_json(self, directory: str, index: int, file_name: str):
        output_path = os.path.join(directory, f"{file_name}.json")

        if not os.path.exists(output_path):
            raise FileNotFoundError(
                f"O arquivo {file_name}.json não foi encontrado no diretório {directory}.")

        with open(output_path, 'r') as file:
            data = json.load(file)

        if index < -1 or index >= len(data):
            raise IndexError(f"Índice {index} fora dos limites. O arquivo contém {len(data)} entradas.")

        result_dict = data[index]

        return result_dict

    def __get_last_project_from_hyper_band(self, hyper_band_executions_directory):
        projects = os.listdir(hyper_band_executions_directory)
        project_names = [p for p in projects if os.path.isdir(os.path.join(hyper_band_executions_directory, p))]
        last_project = project_names[-1]

        return last_project

    def __save_keras_model(self, model):
        model.save(f'{str(model)}.keras')

    def __delete_trials(self):
        pass

