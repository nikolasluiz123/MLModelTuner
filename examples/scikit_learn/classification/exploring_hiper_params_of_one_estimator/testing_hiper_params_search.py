import warnings

import pandas as pd
from scipy.stats import randint, uniform
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from skopt.space import Real, Categorical, Integer

from examples.data.data_processing import get_titanic_train_data
from scikit_learn.features_search.select_k_best_searcher import SelectKBestSearcher
from scikit_learn.hiper_params_search.bayesian_search import BayesianHipperParamsSearcher
from sklearn.experimental import enable_halving_search_cv
from scikit_learn.hiper_params_search.grid_searcher import GridCVHipperParamsSearcher, HalvingGridCVHipperParamsSearcher
from scikit_learn.hiper_params_search.random_searcher import RandomCVHipperParamsSearcher, \
    HalvingRandomCVHipperParamsSearcher
from scikit_learn.history_manager.cross_validator import CrossValidatorHistoryManager
from scikit_learn.process_manager.multi_process_manager import MultiProcessManager
from scikit_learn.process_manager.pipeline import Pipeline
from scikit_learn.validator.cross_validator import CrossValidator

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
#                                    Preparando Comuns para o Modelo Testado                                           #
########################################################################################################################

feature_searcher = SelectKBestSearcher(features_number=5, score_func=f_classif, log_level=1)
validator = CrossValidator(log_level=1)

history_manager = CrossValidatorHistoryManager(output_directory='history',
                                               models_directory='decision_tree_classifier_models',
                                               params_file_name='decision_tree_classifier_best_params',
                                               cv_results_file_name='decision_tree_classifier_cv_results')

best_params_history_manager = CrossValidatorHistoryManager(output_directory='history_bests',
                                                           models_directory='best_models',
                                                           params_file_name='best_params',
                                                           cv_results_file_name='best_params_cv_results')

########################################################################################################################
#                                               Criando os Pipelines                                                   #
########################################################################################################################

pipelines = [
    Pipeline(
        estimator=DecisionTreeClassifier(),
        params={
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'max_depth': randint(1, 10),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
            'min_weight_fraction_leaf': uniform(loc=0.1, scale=0.4),
            'max_features': [None, 'sqrt', 'log2'],
        },
        feature_searcher=feature_searcher,
        params_searcher=RandomCVHipperParamsSearcher(number_iterations=50, log_level=1),
        validator=validator,
        history_manager=history_manager
    ),
    Pipeline(
        estimator=DecisionTreeClassifier(),
        params={
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'max_depth': [2, 4, 6, 8],
            'min_samples_split': [2, 4, 8, 16],
            'min_samples_leaf': [2, 4, 8, 16],
            'min_weight_fraction_leaf': [0.1, 0.3, 0.4],
            'max_features': [None, 'sqrt', 'log2'],
        },
        feature_searcher=feature_searcher,
        params_searcher=HalvingRandomCVHipperParamsSearcher(log_level=1, number_candidates=50),
        validator=validator,
        history_manager=history_manager
    ),
    Pipeline(
        estimator=DecisionTreeClassifier(),
        params={
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'max_depth': [2, 4, 6, 8],
            'min_samples_split': [2, 4, 8, 16],
            'min_samples_leaf': [2, 4, 8, 16],
            'min_weight_fraction_leaf': [0.1, 0.3, 0.4],
            'max_features': [None, 'sqrt', 'log2'],
        },
        feature_searcher=feature_searcher,
        params_searcher=GridCVHipperParamsSearcher(log_level=1),
        validator=validator,
        history_manager=history_manager
    ),
    Pipeline(
        estimator=DecisionTreeClassifier(),
        params={
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'max_depth': [2, 4, 6, 8],
            'min_samples_split': [2, 4, 8, 16],
            'min_samples_leaf': [2, 4, 8, 16],
            'min_weight_fraction_leaf': [0.1, 0.3, 0.4],
            'max_features': [None, 'sqrt', 'log2'],
        },
        feature_searcher=feature_searcher,
        params_searcher=HalvingGridCVHipperParamsSearcher(log_level=1),
        validator=validator,
        history_manager=history_manager
    ),
    Pipeline(
        estimator=DecisionTreeClassifier(),
        params={
            'criterion': Categorical(['gini', 'entropy', 'log_loss']),
            'splitter': Categorical(['best', 'random']),
            'max_depth': Integer(1, 10),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(2, 20),
            'min_weight_fraction_leaf': Real(0.1, 0.5, prior='uniform'),
            'max_features': Categorical([None, 'sqrt', 'log2']),
        },
        feature_searcher=feature_searcher,
        params_searcher=BayesianHipperParamsSearcher(log_level=1, number_iterations=50),
        validator=validator,
        history_manager=history_manager
    ),
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