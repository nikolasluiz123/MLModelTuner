from scipy.stats import randint
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

from examples.data.data_processing import get_workout_train_data
from wrappers.scikit_learn import RandomCVHipperParamsSearcher
from wrappers.scikit_learn import CrossValidatorHistoryManager
from wrappers.scikit_learn import MultiProcessManager
from wrappers.scikit_learn import Pipeline
from wrappers.scikit_learn import CrossValidator

########################################################################################################################
#                                            Preparando os Dados                                                       #
########################################################################################################################

df_train = get_workout_train_data()

label_encoder = LabelEncoder()
df_train['exercicio'] = label_encoder.fit_transform(df_train['exercicio'])

x = df_train.drop(columns=['peso', 'data'])
y = df_train['peso']

########################################################################################################################
#                                    Preparando Implementações que serão Testadas                                      #
########################################################################################################################

params_searcher = RandomCVHipperParamsSearcher(number_iterations=100, log_level=1)

history_manager = CrossValidatorHistoryManager(output_directory='history',
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

pipelines = [
    Pipeline(
        estimator=KNeighborsRegressor(),
        params={
            'n_neighbors': randint(1, 20),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        },
        scaler=StandardScaler(),
        feature_searcher=None,
        params_searcher=params_searcher,
        history_manager=history_manager,
        validator=cross_validator
    ),
    Pipeline(
        estimator=KNeighborsRegressor(),
        params={
            'n_neighbors': randint(1, 20),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        },
        feature_searcher=None,
        params_searcher=params_searcher,
        history_manager=history_manager,
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
    scoring='neg_mean_squared_error',
    save_history=True
)

manager.process_pipelines()
