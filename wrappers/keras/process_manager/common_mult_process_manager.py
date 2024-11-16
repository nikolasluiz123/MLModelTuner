import gc
from abc import abstractmethod

import keras
import pandas as pd
from pandas import DataFrame
from tabulate import tabulate
from tensorflow.python.keras.mixed_precision.policy import set_global_policy

from wrappers.keras.history_manager.common_history_manager import KerasHistoryManager
from wrappers.keras.process_manager.pipeline import KerasPipeline
from wrappers.keras.validator.results.common import KerasValidationResult


class KerasMultiProcessManager:

    def __init__(self,
                 pipelines: list[KerasPipeline] | KerasPipeline,
                 seed: int,
                 history_manager: KerasHistoryManager,
                 policy='mixed_float16',
                 history_index: int | None = None,
                 save_history=True,
                 delete_trials_after_execution=False):
        self.pipelines = pipelines
        self.seed = seed
        self.history_manager = history_manager
        self.history_index = history_index
        self.save_history = save_history
        self.delete_trials_after_execution = delete_trials_after_execution

        self.results = []

        set_global_policy(policy)
        keras.utils.set_random_seed(seed)

    def process_pipelines(self):
        if type(self.pipelines) is list:
            for pipeline in self.pipelines:
                self.__process_single_pipeline(pipeline)
        else:
            self.__process_single_pipeline(self.pipelines)

        df_results = self._show_results()
        self._on_after_process_pipelines(df_results)

    def __process_single_pipeline(self, pipeline: KerasPipeline):
        self.__show_log_init_process(pipeline)

        train_data, validation_data = self.__execute_pre_processing(pipeline)
        model_instance = self.__execute_param_search(train_data, validation_data, pipeline)
        validation_result = self.__execute_validation(pipeline, train_data, validation_data, model_instance)

        self.__save_data_in_history(pipeline, validation_result)
        self._append_new_result(pipeline, validation_result)

        keras.backend.clear_session()
        gc.collect()

    def __execute_param_search(self, train_data, validation_data, pipeline: KerasPipeline):
        if self.history_index is None:
            return pipeline.params_searcher.process(train_data=train_data,
                                                    validation_data=validation_data,
                                                    model=pipeline.model)
        else:
            return None

    def __execute_validation(self, pipeline: KerasPipeline, train_data, validation_data, model_instance) -> KerasValidationResult:
        if model_instance is not None:
            validation_result = pipeline.validator.validate(
                model_instance=model_instance,
                train_data=train_data,
                validation_data=validation_data
            )

            return validation_result
        else:
            return pipeline.history_manager.get_validation_result(self.history_index)

    def __execute_pre_processing(self, pipeline: KerasPipeline) -> tuple:
        if self.history_index is None or not pipeline.history_manager.has_history():
            train_data, validation_data = pipeline.data_pre_processor.get_train_data()
            return train_data, validation_data
        else:
            return None, None

    def __show_log_init_process(self, pipeline: KerasPipeline):
        if self.history_index is None or not pipeline.history_manager.has_history():
            print()
            print('Iniciando o Processamento')

            data = pipeline.get_dict_pipeline_data()
            data = {k: [v] for k, v in data.items()}
            df = pd.DataFrame.from_dict(data, orient='columns')

            print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
            print()

    def _append_new_result(self, pipeline: KerasPipeline, result: KerasValidationResult):
        pipeline_infos = pipeline.get_dict_pipeline_data()
        performance_metrics = result.append_data(pipeline_infos)

        if self.history_index is None or not pipeline.history_manager.has_history():
            self._calculate_processes_time(performance_metrics, pipeline)
        else:
            self._load_processes_time_from_history(performance_metrics, pipeline)

        self.results.append(performance_metrics)

    def __save_data_in_history(self, pipeline: KerasPipeline, validation_result):
        if self.save_history and (self.history_index is None or not pipeline.history_manager.has_history()):
            pre_processing_time, params_search_time, validation_time = pipeline.get_execution_times()

            pipeline.history_manager.save_result(model_instance=validation_result.model,
                                                 model=pipeline.model,
                                                 validation_history=validation_result.history,
                                                 oracle_fields_list=pipeline.params_searcher.get_fields_oracle_json_file(),
                                                 params_search_directory=pipeline.params_searcher.directory,
                                                 params_search_project=pipeline.params_searcher.project_name,
                                                 pre_processing_time=self._format_time(pre_processing_time),
                                                 params_search_time=self._format_time(params_search_time),
                                                 validation_time=self._format_time(validation_time))

    def _calculate_processes_time(self, performance_metrics, pipeline: KerasPipeline):
        pre_processing_time, params_search_time, validation_time = pipeline.get_execution_times()

        performance_metrics['pre_processing_time'] = self._format_time(pre_processing_time)
        performance_metrics['params_search_time'] = self._format_time(params_search_time)
        performance_metrics['validation_time'] = self._format_time(validation_time)

    def _load_processes_time_from_history(self, performance_metrics, pipeline: KerasPipeline):
        history_dict = pipeline.history_manager.get_best_model_executions(self.history_index)

        performance_metrics['pre_processing_time'] = history_dict['pre_processing_time']
        performance_metrics['pre_processing_time'] = history_dict['pre_processing_time']

    @abstractmethod
    def _show_results(self) -> DataFrame:
        ...

    @staticmethod
    def _format_time(seconds):
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)

        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{milliseconds:03}"

    def _on_after_process_pipelines(self, df_results):
        self.__save_best_model(df_results)
        self.__delete_all_pipeline_trials()


    def __save_best_model(self, df_results: DataFrame):
        pipeline_not_executed = self.__get_has_pipeline_not_executed()

        if self.save_history and (self.history_index is None or pipeline_not_executed):
            best = df_results.head(1)

            best_pipeline = self.get_best_pipeline(best)
            executions_history = best_pipeline.history_manager.get_best_model_executions(index=-1)
            model_instance = best_pipeline.history_manager.get_saved_model(best_pipeline.history_manager.get_history_len())

            self.history_manager.save_result(model_instance=model_instance,
                                             model=best_pipeline.model,
                                             validation_history=executions_history,
                                             params_search_directory=best_pipeline.params_searcher.directory,
                                             params_search_project=best_pipeline.params_searcher.project_name,
                                             oracle_fields_list=best_pipeline.params_searcher.get_fields_oracle_json_file(),
                                             pre_processing_time=executions_history['pre_processing_time'],
                                             params_search_time=executions_history['params_search_time'],
                                             validation_time=executions_history['validation_time'])

    def __get_has_pipeline_not_executed(self):
        pipeline_not_executed = False
        if type(self.pipelines) is list:
            for p in self.pipelines:
                if not p.history_manager.has_history():
                    pipeline_not_executed = True
        else:
            if not self.pipelines.history_manager.has_history():
                pipeline_not_executed = True

        return pipeline_not_executed

    def get_best_pipeline(self, best) -> KerasPipeline:
        if type(self.pipelines) is list:
            best_pipeline = [pipe for pipe in self.pipelines if self.__is_best_pipeline(best, pipe)][0]
        else:
            best_pipeline = self.pipelines

        return best_pipeline

    def __is_best_pipeline(self, df: DataFrame, pipe: KerasPipeline):
        return (
                df['model'].values[0] == type(pipe.model).__name__ and
                df['data_pre_processor'].values[0] == type(pipe.data_pre_processor).__name__ and
                df['params_searcher'].values[0] == type(pipe.params_searcher).__name__ and
                df['searcher_objective'].values[0] == pipe.params_searcher.objective and
                df['searcher_epochs'].values[0] == pipe.params_searcher.epochs and
                df['searcher_batch_size'].values[0] == pipe.params_searcher.batch_size and
                df['project_name'].values[0] == pipe.params_searcher.project_name and
                df['validator'].values[0] == type(pipe.validator).__name__ and
                df['validator_epochs'].values[0] == pipe.validator.epochs and
                df['validator_batch_size'].values[0] == pipe.validator.batch_size
        )

    def __delete_all_pipeline_trials(self):
        if self.history_index is None and self.delete_trials_after_execution:
            if type(self.pipelines) is list:
                for pipeline in self.pipelines:
                    pipeline.history_manager.delete_trials(pipeline.params_searcher.directory,
                                                           pipeline.params_searcher.project_name)
            else:
                self.pipelines.history_manager.delete_trials(self.pipelines.params_searcher.directory,
                                                             self.pipelines.params_searcher.project_name)
