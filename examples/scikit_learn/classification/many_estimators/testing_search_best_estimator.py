import warnings

import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from examples.data.data_processing import get_train_data
from scikit_learn.features_search.select_k_best_searcher import SelectKBestSearcher
from scikit_learn.hiper_params_search.random_searcher import RandomCVHipperParamsSearcher
from scikit_learn.history_manager.cross_validator import CrossValidatorHistoryManager
from scikit_learn.process_manager.multi_process_manager import MultiProcessManager
from scikit_learn.process_manager.pipeline import Pipeline
from scikit_learn.validator.cross_validator import CrossValidator

warnings.filterwarnings("ignore", category=RuntimeWarning)

########################################################################################################################
#                                            Preparando os Dados                                                       #
########################################################################################################################
df_train = get_train_data()

x = df_train.drop(columns=['sobreviveu'], axis=1)

obj_columns = df_train.select_dtypes(include='object').columns

x = pd.get_dummies(x, columns=obj_columns)
y = df_train['sobreviveu']

########################################################################################################################
#                                    Preparando Comuns para os Modelos Testados                                        #
########################################################################################################################

feature_searcher = SelectKBestSearcher(features_number=5, score_func=f_classif, log_level=1)
params_searcher = RandomCVHipperParamsSearcher(number_iterations=50, log_level=1)
validator = CrossValidator(log_level=1)

best_params_history_manager = CrossValidatorHistoryManager(output_directory='history_bests',
                                                           models_directory='best_models',
                                                           params_file_name='best_params')

########################################################################################################################
#                           Criando os Pipelines com os Modelos e suas Especificidades                                 #
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
        params_searcher=params_searcher,
        validator=validator,
        history_manager=CrossValidatorHistoryManager(
            output_directory='history',
            models_directory='decision_tree_classifier_models',
            params_file_name='decision_tree_classifier_best_params')
    ),
    Pipeline(
        estimator=RandomForestClassifier(),
        params={
            'n_estimators': randint(10, 50),
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': randint(1, 20),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
            'min_weight_fraction_leaf': uniform(loc=0.1, scale=0.4),
            'max_features': [None, 'sqrt', 'log2']
        },
        feature_searcher=feature_searcher,
        params_searcher=params_searcher,
        validator=validator,
        history_manager=CrossValidatorHistoryManager(
            output_directory='history',
            models_directory='random_forest_classifier_models',
            params_file_name='random_forest_classifier_best_params')
    ),
    Pipeline(
        estimator=GaussianProcessClassifier(),
        params={
            'optimizer': ['fmin_l_bfgs_b', None],
            'n_restarts_optimizer': randint(0, 10),
            'multi_class': ['one_vs_rest', 'one_vs_one'],
            'max_iter_predict': [100, 300, 500, 700, 900]
        },
        feature_searcher=feature_searcher,
        params_searcher=params_searcher,
        validator=validator,
        history_manager=CrossValidatorHistoryManager(
            output_directory='history',
            models_directory='gausian_process_classifier_models',
            params_file_name='gausian_process_classifier_best_params')
    ),
    Pipeline(
        estimator=KNeighborsClassifier(),
        params={
            'n_neighbors': randint(1, 10),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': randint(1, 100),
            'p': [1, 2],
            'metric': ['minkowski', 'euclidean', 'manhattan']
        },
        feature_searcher=feature_searcher,
        params_searcher=params_searcher,
        validator=validator,
        history_manager=CrossValidatorHistoryManager(
            output_directory='history',
            models_directory='k_neighbors_classifier_models',
            params_file_name='k_neighbors_classifier_best_params')
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