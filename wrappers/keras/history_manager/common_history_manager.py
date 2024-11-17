import json
import os
import shutil
from abc import ABC, abstractmethod
from typing import TypeVar

import keras

from wrappers.common.history_manager.common_history_manager import CommonHistoryManager
from wrappers.keras.validator.results.common_validation_result import KerasValidationResult

KerasValResult = TypeVar('KerasValResult', bound=KerasValidationResult)

class KerasHistoryManager(CommonHistoryManager[KerasValResult], ABC):

    def __init__(self,
                 output_directory: str,
                 models_directory: str,
                 best_params_file_name: str):
        super().__init__(output_directory, models_directory, best_params_file_name)

    def save_result(self,
                    model,
                    model_instance,
                    validation_history,
                    params_search_directory: str,
                    params_search_project: str,
                    oracle_fields_list: list[str],
                    pre_processing_time: str,
                    params_search_time: str,
                    validation_time: str):
        oracle_data = self.__get_oracle_file(params_search_directory, params_search_project)

        best_model_execution_data = {
            'model': type(model).__name__,
            'history': validation_history,
            'neural_structure': oracle_data['hyperparameters'],
        }

        for field in oracle_fields_list:
            best_model_execution_data[field] = oracle_data[field]

        best_model_execution_data['pre_processing_time'] = pre_processing_time
        best_model_execution_data['params_search_time'] = params_search_time
        best_model_execution_data['validation_time'] = validation_time

        self._save_dictionary_in_json(best_model_execution_data, file_name=self.best_params_file_name)
        self._save_model(model_instance)

    def _save_model(self, model):
        history_len = self.get_history_len()
        path = os.path.join(self.models_directory, f"model_{history_len}.keras")
        model.save(path)

    def get_saved_model(self, version: int):
        output_path = os.path.join(self.models_directory, f"model_{version}.keras")

        if not os.path.exists(output_path):
            raise FileNotFoundError(
                f"O modelo versão {version} não foi encontrado no diretório {self.models_directory}.")

        return keras.models.load_model(output_path)

    def __get_oracle_file(self, search_directory: str, project_name: str):
        oracle_file_path = os.path.join(search_directory, project_name, 'oracle.json')

        if os.path.isfile(oracle_file_path):
            with open(oracle_file_path, 'r') as file:
                oracle_data = json.load(file)
        else:
            print(f'O arquivo "oracle.json" não foi encontrado no diretório {oracle_file_path}.')

        return oracle_data

    def delete_trials(self, directory: str, project_name: str):
        trials_path = os.path.join(directory, project_name)
        shutil.rmtree(trials_path)

        print(f'Diretório "{trials_path}" e todos os arquivos/subdiretórios foram removidos com sucesso.')
