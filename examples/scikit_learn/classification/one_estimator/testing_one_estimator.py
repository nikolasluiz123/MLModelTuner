import pandas as pd
from scipy.stats import randint, uniform
from sklearn.tree import DecisionTreeClassifier

from examples.data.data_processing import get_titanic_train_data
from scikit_learn.features_search.rfe_searcher import RecursiveFeatureCVSearcher
from scikit_learn.hiper_params_search.random_searcher import RandomCVHipperParamsSearcher
from scikit_learn.history_manager.cross_validator import CrossValidatorHistoryManager
from scikit_learn.process_manager.multi_process_manager import MultiProcessManager
from scikit_learn.process_manager.pipeline import Pipeline
from scikit_learn.validator.cross_validator import CrossValidator

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

random_cv_hyper_params_searcher = RandomCVHipperParamsSearcher(number_iterations=500, log_level=1)

cross_validator_history_manager = CrossValidatorHistoryManager(output_directory='history',
                                                               models_directory='models',
                                                               params_file_name='params',
                                                               cv_results_file_name='cv_results')

best_params_history_manager = CrossValidatorHistoryManager(output_directory='history_bests',
                                                           models_directory='best_models',
                                                           params_file_name='best_params',
                                                           cv_results_file_name='best_cv_results')
cross_validator = CrossValidator(log_level=1)

########################################################################################################################
#                                               Criando o Pipeline                                                     #
########################################################################################################################

pipeline = Pipeline(
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
    feature_searcher=recursive_feature_cv_searcher,
    params_searcher=random_cv_hyper_params_searcher,
    history_manager=cross_validator_history_manager,
    validator=cross_validator
)

########################################################################################################################
#                                      Criando e Executando o Process Manager                                          #
########################################################################################################################

manager = MultiProcessManager(
    data_x=x,
    data_y=y,
    seed=42,
    pipelines=pipeline,
    fold_splits=5,
    history_manager=best_params_history_manager,
    scoring='accuracy',
    stratified=True,
    save_history=True
)

manager.process_pipelines()