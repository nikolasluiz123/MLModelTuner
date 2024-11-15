import pandas as pd
from pandas import DataFrame
from tabulate import tabulate

from wrappers.keras.process_manager.common_mult_process_manager import KerasMultiProcessManager


class KerasClassifierMultProcessManager(KerasMultiProcessManager):

    def _show_results(self) -> DataFrame:
        df_results = pd.DataFrame(self.results)
        df_results = df_results.sort_values(
            by=['mean_val_accuracy', 'standard_deviation_val_accuracy', 'mean_val_loss', 'standard_deviation_val_loss'],
            ascending=[False, True, False, True])

        print(tabulate(df_results, headers='keys', tablefmt='fancy_grid', floatfmt=".6f", showindex=False))

        return df_results