import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from wrappers.common.data_pre_processor.common_data_pre_processor import CommonDataPreProcessor


class ScikitLearnWorkoutPreProcessorExample(CommonDataPreProcessor):
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
            df_train = self.get_workout_train_data()

            label_encoder = LabelEncoder()
            df_train['exercicio'] = label_encoder.fit_transform(df_train['exercicio'])

            x = df_train.drop(columns=['peso', 'data'])
            y = df_train['peso']

            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

            self._data_split = (x_train, x_val, y_train, y_val)

        return self._data_split

    @staticmethod
    def get_workout_train_data() -> DataFrame:
        """
        Carrega e pré-processa os dados de treino para exercícios físicos.
        """
        df = pd.read_csv('../data/workout_train_data.csv')
        df.drop(columns=['Distance', 'Seconds', 'Notes', 'Workout Notes', 'Workout Name'], inplace=True, axis=1)
        df.rename(
            columns={
                'Date': 'data',
                'Exercise Name': 'exercicio',
                'Set Order': 'serie',
                'Weight': 'peso',
                'Reps': 'repeticoes'
            },
            inplace=True
        )
        df['peso'] = df['peso'] * 0.453592
        df['data'] = pd.to_datetime(df['data'])
        df.drop(index=df[(df['peso'] == 0) | (df['repeticoes'] == 0)].index, inplace=True)

        return df
