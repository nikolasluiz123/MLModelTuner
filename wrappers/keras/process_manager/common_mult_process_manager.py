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
                 save_history=True):
        self.pipelines = pipelines
        self.seed = seed
        self.history_manager = history_manager
        self.history_index = history_index
        self.save_history = save_history

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
        validation_result = self.__execute_validation(pipeline, train_data, validation_data)

        self.__save_data_in_history(pipeline, validation_result)
        self._append_new_result(pipeline, validation_result)

        keras.backend.clear_session()
        gc.collect()

    def __execute_validation(self, pipeline: KerasPipeline, train_data, validation_data) -> KerasValidationResult:
        if self.history_index is None:
            validation_result = pipeline.validator.process(
                train_data=train_data,
                validation_data=validation_data,
                model=pipeline.model,
                project_name=pipeline.hyper_band_config.project_name,
                hyper_band_config=pipeline.hyper_band_config,
                search_config=pipeline.search_config,
                final_fit_config=pipeline.final_fit_config
            )

            return validation_result
        else:
            return pipeline.history_manager.get_validation_result(self.history_index)

    def __execute_pre_processing(self, pipeline: KerasPipeline) -> tuple:
        if self.history_index is None:
            train_data, validation_data = pipeline.data_pre_processor.get_train_data()
            return train_data, validation_data
        else:
            return None, None

    def __show_log_init_process(self, pipeline: KerasPipeline):
        if self.history_index is None:
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

        if self.history_index is None:
            self._calculate_processes_time(performance_metrics, pipeline)
        else:
            self._load_processes_time_from_history(performance_metrics, pipeline)

        self.results.append(performance_metrics)

    def __save_data_in_history(self, pipeline: KerasPipeline, validation_result):
        if self.save_history and self.history_index is None:
            pre_processing_time, validation_time = pipeline.get_execution_times()

            pipeline.history_manager.save_result(model_instance=validation_result.model,
                                                 model=pipeline.model,
                                                 final_fit_history=validation_result.history,
                                                 hyper_band_executions_directory=pipeline.hyper_band_config.directory,
                                                 pre_processing_time=self._format_time(pre_processing_time),
                                                 validation_time=self._format_time(validation_time))

    def _calculate_processes_time(self, performance_metrics, pipeline: KerasPipeline):
        pre_processing_time, validation_time = pipeline.get_execution_times()

        performance_metrics['pre_processing_time'] = self._format_time(pre_processing_time)
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
        if self.save_history and self.history_index is None:
            best = df_results.head(1)

            best_pipeline = self.get_best_pipeline(best)
            executions_history = best_pipeline.history_manager.get_best_model_executions(index=-1)
            model_instance = best_pipeline.history_manager.get_saved_model(best_pipeline.history_manager.get_history_len())

            self.history_manager.save_result(model_instance=model_instance,
                                             model=best_pipeline.model,
                                             final_fit_history=executions_history,
                                             hyper_band_executions_directory=best_pipeline.hyper_band_config.directory,
                                             pre_processing_time=executions_history['pre_processing_time'],
                                             validation_time=executions_history['validation_time'])

    def get_best_pipeline(self, best):
        if type(self.pipelines) is list:
            best_pipeline = [pipe for pipe in self.pipelines if self.__is_best_pipeline(best, pipe)][0]
        else:
            best_pipeline = self.pipelines

        return best_pipeline

    def __is_best_pipeline(self, df: DataFrame, pipe: KerasPipeline):
        return (
                df['model'].values[0] == type(pipe.model).__name__ and
                df['validator'].values[0] == type(pipe.validator).__name__ and
                df['objective'].values[0] == str(pipe.hyper_band_config.objective) and
                df['factor'].values[0] == str(pipe.hyper_band_config.factor) and
                df['max_epochs'].values[0] == str(pipe.hyper_band_config.max_epochs) and
                df['search_epochs'].values[0] == str(pipe.search_config.epochs) and
                df['search_batch_size'].values[0] == str(pipe.search_config.batch_size) and
                df['search_callbacks'].values[0] == [type(c).__name__ for c in pipe.search_config.callbacks] and
                df['final_fit_epochs'].values[0] == str(pipe.final_fit_config.epochs) and
                df['final_fit_batch_size'].values[0] == str(pipe.final_fit_config.batch_size)
        )

    def __delete_all_pipeline_trials(self):
        if self.history_index is None:
            if type(self.pipelines) is list:
                for pipeline in self.pipelines:
                    pipeline.history_manager.delete_trials(pipeline.hyper_band_config)
            else:
                self.pipelines.history_manager.delete_trials(self.pipelines.hyper_band_config)
