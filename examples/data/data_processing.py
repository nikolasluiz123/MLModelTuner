import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame


def get_titanic_train_data() -> DataFrame:
    df = pd.read_csv(r'C:\Users\nikol\git\IA\MLModelTunner\examples\data\titanic_train_data.csv')

    df.columns = ['id_passageiro', 'sobreviveu', 'classe_social', 'nome', 'sexo', 'idade', 'qtd_irmaos_conjuges',
                  'qtd_pais_filhos', 'ticket', 'valor_ticket', 'cabine', 'porta_embarque']

    df.drop(columns=['id_passageiro', 'nome', 'ticket', 'valor_ticket', 'cabine'], inplace=True, axis=1)
    df.dropna(subset=['idade'], inplace=True)

    return df

def get_workout_train_data() -> DataFrame:
    df = pd.read_csv(r'C:\Users\nikol\git\IA\MLModelTunner\examples\data\workout_train_data.csv')
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

