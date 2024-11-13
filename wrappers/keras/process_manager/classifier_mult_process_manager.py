import pandas as pd
from pandas import DataFrame
from tabulate import tabulate

from wrappers.keras.process_manager.common_mult_process_manager import KerasMultiProcessManager


class KerasClassifierMultProcessManager(KerasMultiProcessManager):

    def _show_results(self) -> DataFrame:
        df_results = pd.DataFrame(self.results)
        df_results = df_results.sort_values(by=['val_accuracy', 'val_loss'], ascending=False)

        print(tabulate(df_results, headers='keys', tablefmt='fancy_grid', floatfmt=".6f", showindex=False))

        return df_results