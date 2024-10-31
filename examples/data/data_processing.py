import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame


def get_train_data() -> DataFrame:
    df = pd.read_csv(r'C:\Users\nikol\git\IA\MLModelTunner\examples\data\train.csv')

    df.columns = ['id_passageiro', 'sobreviveu', 'classe_social', 'nome', 'sexo', 'idade', 'qtd_irmaos_conjuges',
                  'qtd_pais_filhos', 'ticket', 'valor_ticket', 'cabine', 'porta_embarque']

    df.drop(columns=['id_passageiro', 'nome', 'ticket', 'valor_ticket', 'cabine'], inplace=True, axis=1)
    df.dropna(subset=['idade'], inplace=True)

    return df
