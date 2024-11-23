import pandas as pd
from pandas import DataFrame
from tabulate import tabulate

from wrappers.keras.process_manager.common_multi_process_manager import KerasMultiProcessManager


class KerasRegressorMultProcessManager(KerasMultiProcessManager):
    """
     Implementação de gerenciador de processos específica para regressão, isso é necessário devido aos campos retornados
     na validação desse tipo de modelo de rede reural.
     """

    def _show_results(self) -> DataFrame:
        df_results = pd.DataFrame(self.results)
        df_results = df_results.sort_values(
            by=['mean_absolute_error', 'standard_deviation_absolute_error', 'mean_val_loss', 'standard_deviation_val_loss'],
            ascending=[True, True, True, True]
        )

        print(tabulate(df_results, headers='keys', tablefmt='fancy_grid', floatfmt=".6f", showindex=False))

        return df_results