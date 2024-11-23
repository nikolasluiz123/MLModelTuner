import pandas as pd
from sklearn.preprocessing import StandardScaler

from wrappers.scikit_learn.history_manager.common_history_manager import ScikitLearnCommonHistoryManager
from wrappers.scikit_learn.validator.results.cross_validation_result import ScikitLearnCrossValidationResult


class ScikitLearnCrossValidationHistoryManager(ScikitLearnCommonHistoryManager[ScikitLearnCrossValidationResult]):
    """
    Implementação de histórico específica para armazenar dados obtidos da validação cruzada.

    Por conta desse tipo de validação fornecer muitos dados adicionais é preciso uma implementação específica de manutenção
    do histórico.
    """

    def __init__(self, output_directory: str, models_directory: str, best_params_file_name: str, cv_results_file_name: str):
        super().__init__(output_directory, models_directory, best_params_file_name, cv_results_file_name)

    def save_result(self,
                    validation_result: ScikitLearnCrossValidationResult,
                    cv_results,
                    pre_processing_time: str,
                    feature_selection_time: str,
                    search_time: str,
                    validation_time: str,
                    scoring: str,
                    features: list[str]):
        dictionary_execution_infos = {
            'estimator': type(validation_result.estimator).__name__,
            'mean': validation_result.mean,
            'standard_deviation': validation_result.standard_deviation,
            'median': validation_result.median,
            'variance': validation_result.variance,
            'standard_error': validation_result.standard_error,
            'min_max_score': validation_result.min_max_score,
            'estimator_params': validation_result.estimator.get_params(),
            'scoring': scoring,
            'features': ", ".join(features),
            'pre_processing_time': pre_processing_time,
            'feature_selection_time': feature_selection_time,
            'search_time': search_time,
            'validation_time': validation_time
        }

        self._save_dictionary_in_json(dictionary_execution_infos, file_name=self.best_params_file_name)
        self._save_cv_results(cv_results)
        self._save_model(validation_result.estimator)

    def _save_cv_results(self, cv_results):
        """
        Função para armazenar as combinações de parâmetros testadas na busca em um JSON.
        """

        df = pd.DataFrame.from_dict(cv_results, orient='columns')
        self._save_dictionary_in_json(df.to_dict(), file_name=self.cv_results_file_name)

    def load_validation_result_from_history(self, index: int = -1) -> ScikitLearnCrossValidationResult:
        result_dict = self.get_dictionary_from_params_json(index)

        return ScikitLearnCrossValidationResult(
            mean=result_dict['mean'],
            standard_deviation=result_dict['standard_deviation'],
            median=result_dict['median'],
            variance=result_dict['variance'],
            standard_error=result_dict['standard_error'],
            min_max_score=result_dict['min_max_score'],
            scoring=result_dict['scoring'],
            estimator=self.get_saved_model(self.get_history_len())
        )
