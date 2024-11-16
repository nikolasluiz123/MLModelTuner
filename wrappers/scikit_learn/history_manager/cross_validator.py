import pandas as pd
from sklearn.preprocessing import StandardScaler

from wrappers.scikit_learn.history_manager.common_history_manager import ScikitLearnCommonHistoryManager
from wrappers.scikit_learn.validator.results.cross_validation import ScikitLearnCrossValidationResult


class CrossValidatorScikitLearnCommonHistoryManager(ScikitLearnCommonHistoryManager[ScikitLearnCrossValidationResult]):
    """
    Classe que gerencia o histórico de validação cruzada dos modelos.

    Esta classe estende a classe HistoryManager para implementar o armazenamento específico
    dos resultados de validação cruzada. Os resultados são salvos em formato JSON e o modelo treinado
    é armazenado usando pickle, permitindo o uso posterior do modelo.
    """

    def __init__(self, output_directory: str, models_directory: str, params_file_name: str, cv_results_file_name: str):
        super().__init__(output_directory, models_directory, params_file_name, cv_results_file_name)

    def save_result(self,
                    classifier_result: ScikitLearnCrossValidationResult,
                    cv_results,
                    feature_selection_time: str,
                    search_time: str,
                    validation_time: str,
                    scoring: str,
                    features: list[str],
                    scaler: StandardScaler | None):
        dictionary_execution_infos = {
            'estimator': type(classifier_result.estimator).__name__,
            'mean': classifier_result.mean,
            'standard_deviation': classifier_result.standard_deviation,
            'median': classifier_result.median,
            'variance': classifier_result.variance,
            'standard_error': classifier_result.standard_error,
            'min_max_score': classifier_result.min_max_score,
            'estimator_params': classifier_result.estimator.get_params(),
            'scoring': scoring,
            'features': ", ".join(features),
            'feature_selection_time': feature_selection_time,
            'search_time': search_time,
            'validation_time': validation_time,
            'scaler': type(scaler).__name__
        }

        self._create_output_dir()
        self._save_dictionary_in_json(dictionary_execution_infos, file_name=self.params_file_name)
        self._save_cv_results(cv_results)
        self._save_model(classifier_result.estimator)

    def _save_cv_results(self, cv_results):
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
            estimator=self.get_saved_model(self._get_history_len())
        )
