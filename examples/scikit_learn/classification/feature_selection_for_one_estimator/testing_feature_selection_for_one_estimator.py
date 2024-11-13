import warnings

import pandas as pd
from scipy.stats import randint, uniform
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier

from examples.data.data_processing import get_titanic_train_data
from wrappers.scikit_learn import GenericUnivariateSelectSearcher
from wrappers.scikit_learn import RecursiveFeatureCVSearcher, RecursiveFeatureSearcher
from wrappers.scikit_learn import SelectKBestSearcher
from wrappers.scikit_learn import SelectPercentileSearcher
from wrappers.scikit_learn import SequentialFeatureSearcher
from wrappers.scikit_learn import RandomCVHipperParamsSearcher
from wrappers.scikit_learn import CrossValidatorHistoryManager
from wrappers.scikit_learn import MultiProcessManager
from wrappers.scikit_learn import Pipeline
from wrappers.scikit_learn import CrossValidator

warnings.filterwarnings("ignore", category=RuntimeWarning)

########################################################################################################################
#                                            Preparando os Dados                                                       #
########################################################################################################################
df_train = get_titanic_train_data()

x = df_train.drop(columns=['sobreviveu'], axis=1)

obj_columns = df_train.select_dtypes(include='object').columns

x = pd.get_dummies(x, columns=obj_columns)
y = df_train['sobreviveu']

########################################################################################################################
#                                    Preparando Implementações que serão Testadas                                      #
########################################################################################################################

recursive_feature_cv_searcher = RecursiveFeatureCVSearcher(log_level=1)
recursive_feature_searcher = RecursiveFeatureSearcher(features_number=5, log_level=1)
select_k_best_searcher = SelectKBestSearcher(features_number=5, score_func=f_classif, log_level=1)
generic_univariate_select_searcher_kbest = GenericUnivariateSelectSearcher(score_func=f_classif, mode='k_best', mode_param=5, log_level=1)
generic_univariate_select_searcher_percentil = GenericUnivariateSelectSearcher(score_func=f_classif, mode='percentile', mode_param=5, log_level=1)
select_percentile_searcher = SelectPercentileSearcher(percent=10, score_func=f_classif, log_level=1)
sequential_feature_searcher = SequentialFeatureSearcher(number_features=5, log_level=1)

random_cv_hyper_params_searcher = RandomCVHipperParamsSearcher(number_iterations=50, log_level=1)

cross_validator_history_manager = CrossValidatorHistoryManager(output_directory='history',
                                                               models_directory='models',
                                                               params_file_name='params',
                                                               cv_results_file_name='cv_results')

best_params_history_manager = CrossValidatorHistoryManager(output_directory='history_bests',
                                                           models_directory='best_models',
                                                           params_file_name='best_params',
                                                           cv_results_file_name='best_cv_results')
cross_validator = CrossValidator(log_level=1)

estimator_params = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': randint(1, 10),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'min_weight_fraction_leaf': uniform(loc=0.1, scale=0.4),
    'max_features': [None, 'sqrt', 'log2'],
}

########################################################################################################################
#                                               Criando os Pipelines                                                   #
########################################################################################################################

pipelines = [
    Pipeline(
        estimator=DecisionTreeClassifier(),
        params=estimator_params,
        feature_searcher=recursive_feature_cv_searcher,
        params_searcher=random_cv_hyper_params_searcher,
        history_manager=cross_validator_history_manager,
        validator=cross_validator
    ),
    Pipeline(
        estimator=DecisionTreeClassifier(),
        params=estimator_params,
        feature_searcher=recursive_feature_searcher,
        params_searcher=random_cv_hyper_params_searcher,
        history_manager=cross_validator_history_manager,
        validator=cross_validator
    ),
    Pipeline(
        estimator=DecisionTreeClassifier(),
        params=estimator_params,
        feature_searcher=select_k_best_searcher,
        params_searcher=random_cv_hyper_params_searcher,
        history_manager=cross_validator_history_manager,
        validator=cross_validator
    ),
    Pipeline(
        estimator=DecisionTreeClassifier(),
        params=estimator_params,
        feature_searcher=generic_univariate_select_searcher_kbest,
        params_searcher=random_cv_hyper_params_searcher,
        history_manager=cross_validator_history_manager,
        validator=cross_validator
    ),
    Pipeline(
        estimator=DecisionTreeClassifier(),
        params=estimator_params,
        feature_searcher=generic_univariate_select_searcher_percentil,
        params_searcher=random_cv_hyper_params_searcher,
        history_manager=cross_validator_history_manager,
        validator=cross_validator
    ),
    Pipeline(
        estimator=DecisionTreeClassifier(),
        params=estimator_params,
        feature_searcher=select_percentile_searcher,
        params_searcher=random_cv_hyper_params_searcher,
        history_manager=cross_validator_history_manager,
        validator=cross_validator
    ),
    Pipeline(
        estimator=DecisionTreeClassifier(),
        params=estimator_params,
        feature_searcher=sequential_feature_searcher,
        params_searcher=random_cv_hyper_params_searcher,
        history_manager=cross_validator_history_manager,
        validator=cross_validator
    )
]

########################################################################################################################
#                                      Criando e Executando o Process Manager                                          #
########################################################################################################################

manager = MultiProcessManager(
    data_x=x,
    data_y=y,
    seed=42,
    pipelines=pipelines,
    fold_splits=5,
    history_manager=best_params_history_manager,
    scoring='accuracy',
    stratified=True,
    save_history=True
)

manager.process_pipelines()
