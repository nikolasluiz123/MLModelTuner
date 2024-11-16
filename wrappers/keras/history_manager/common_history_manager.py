import json
import os
import shutil
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import keras

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

    @abstractmethod
    def get_validation_result(self, index: int) -> Result:
        ...

    def _create_output_dir(self):
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        if not os.path.exists(self.models_directory):
            os.makedirs(self.models_directory)

    def save_result(self,
                    model,
                    model_instance,
                    validation_history,
                    params_search_directory: str,
                    oracle_fields_list: list[str],
                    pre_processing_time: str,
                    params_search_time: str,
                    validation_time: str):
        last_project = self.__get_last_project_from_hyper_band(params_search_directory)
        oracle_data = self.__get_oracle_file(params_search_directory, last_project)

        self.__save_best_model_execution_data(model,
                                              validation_history,
                                              oracle_data,
                                              oracle_fields_list,
                                              pre_processing_time,
                                              params_search_time,
                                              validation_time)
        self.__save_keras_model(model_instance)

    def __save_best_model_execution_data(self,
                                         model,
                                         final_fit_history,
                                         oracle_data,
                                         oracle_fields_list: list[str],
                                         pre_processing_time: str,
                                         params_search_time: str,
                                         validation_time: str):
        best_model_execution_data = {
            'model': type(model).__name__,
            'history': final_fit_history,
            'neural_structure': oracle_data['hyperparameters'],
        }

        for field in oracle_fields_list:
            best_model_execution_data[field] = oracle_data[field]

        best_model_execution_data['pre_processing_time'] = pre_processing_time
        best_model_execution_data['params_search_time'] = params_search_time
        best_model_execution_data['validation_time'] = validation_time

        output_path = os.path.join(self.output_directory, f"{self.best_executions_file_name}.json")

        if os.path.exists(output_path):
            with open(output_path, 'r') as file:
                data = json.load(file)
        else:
            data = []

        data.append(best_model_execution_data)

        with open(output_path, 'w') as file:
            json.dump(data, file, indent=4)

    def get_best_model_executions(self, index: int):
        executions = self._get_dictionary_from_json(self.output_directory, index, self.best_executions_file_name)

        return executions

    def __get_oracle_file(self, hyper_band_executions_directory, last_project):
        oracle_file_path = os.path.join(hyper_band_executions_directory, last_project, 'oracle.json')

        if os.path.isfile(oracle_file_path):
            with open(oracle_file_path, 'r') as file:
                oracle_data = json.load(file)
        else:
            print(f'O arquivo "oracle.json" não foi encontrado no diretório {hyper_band_executions_directory}.')

        return oracle_data

    def _get_dictionary_from_json(self, directory: str, index: int, file_name: str):
        output_path = os.path.join(directory, f"{file_name}.json")

        if not os.path.exists(output_path):
            raise FileNotFoundError(
                f"O arquivo {file_name}.json não foi encontrado no diretório {directory}.")

        with open(output_path, 'r') as file:
            data = json.load(file)

        # if index < -1 or index >= len(data):
        #     raise IndexError(f"Índice {index} fora dos limites. O arquivo contém {len(data)} entradas.")

        result_dict = data[index]

        return result_dict

    def __get_last_project_from_hyper_band(self, hyper_band_executions_directory):
        projects = os.listdir(hyper_band_executions_directory)
        project_names = [p for p in projects if os.path.isdir(os.path.join(hyper_band_executions_directory, p))]
        last_project = project_names[-1]

        return last_project

    def __save_keras_model(self, model):
        path = os.path.join(self.models_directory, f"model_{self.get_history_len()}.keras")
        model.save(path)

    def get_history_len(self) -> int:
        output_path = os.path.join(self.output_directory, f"{self.best_executions_file_name}.json")

        if not os.path.exists(output_path):
            return 0

        with open(output_path, 'r') as file:
            data = json.load(file)
            return len(data)

    def delete_trials(self, directory: str, project_name: str):
        trials_path = os.path.join(directory, project_name)
        shutil.rmtree(trials_path)

        print(f'Diretório "{trials_path}" e todos os arquivos/subdiretórios foram removidos com sucesso.')

    def has_history(self) -> bool:
        output_path = os.path.join(self.output_directory, f"{self.best_executions_file_name}.json")

        if not os.path.exists(output_path):
            return False

        with open(output_path, 'r') as file:
            data = json.load(file)
            return len(data) > 0

    def get_saved_model(self, version: int):
        output_path = os.path.join(self.models_directory, f"model_{version}.keras")

        if not os.path.exists(output_path):
            raise FileNotFoundError(
                f"O modelo versão {version} não foi encontrado no diretório {self.models_directory}.")

        return keras.models.load_model(output_path)
