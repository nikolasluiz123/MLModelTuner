import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from wrappers.common.data_pre_processor.common_data_pre_processor import CommonDataPreProcessor


class ScikitLearnTitanicPreProcessorExample(CommonDataPreProcessor):
    def __init__(self):
        super().__init__()
        self._data_split = None

    def _on_execute_train_process(self) -> tuple:
        """
        Processa os dados de treino, separando 80% dos dados para treinamento.
        """
        x_train, x_val, y_train, y_val = self._get_data_split()

        return x_train, y_train

    def get_data_additional_validation(self) -> tuple:
        """
        Retorna os 20% dos dados de validação, independentemente da execução do processo de treino.
        """
        _, x_val, _, y_val = self._get_data_split()

        return x_val, y_val

    def _get_data_split(self) -> tuple:
        """
        Realiza a divisão dos dados de treino e validação uma única vez.
        """
        if self._data_split is None:
            df_train = self.__get_titanic_train_data()

            x = df_train.drop(columns=['sobreviveu'], axis=1)
            obj_columns = df_train.select_dtypes(include='object').columns

            x = pd.get_dummies(x, columns=obj_columns)
            y = df_train['sobreviveu']

            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

            self._data_split = (x_train, x_val, y_train, y_val)

        return self._data_split

    @staticmethod
    def __get_titanic_train_data() -> DataFrame:
        """
        Carrega e pré-processa os dados do Titanic.
        """
        df = pd.read_csv('../data/titanic_train_data.csv')

        df.columns = ['id_passageiro', 'sobreviveu', 'classe_social', 'nome', 'sexo', 'idade', 'qtd_irmaos_conjuges',
                      'qtd_pais_filhos', 'ticket', 'valor_ticket', 'cabine', 'porta_embarque']

        df.drop(columns=['id_passageiro', 'nome', 'ticket', 'valor_ticket', 'cabine'], inplace=True, axis=1)
        df.dropna(subset=['idade'], inplace=True)

        return df
