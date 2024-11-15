import math

from keras.src.callbacks import Callback


class HyperBandConfig:

    def __init__(self,
               objective: str | list[str],
               factor: int,
               max_epochs: int,
               directory: str,
               project_name: str):
        self.objective = objective
        self.factor = factor
        self.max_epochs = max_epochs
        self.directory = directory
        self.project_name = project_name

class SearchConfig:

    def __init__(self,
                 epochs: int,
                 batch_size: int,
                 callbacks: list[Callback],
                 folds: int,
                 stratified: bool,
                 log_level: int):
        self.epochs = epochs
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.folds = folds
        self.stratified = stratified
        self.log_level = log_level

class FinalFitConfig:

    def __init__(self,
                 epochs: int,
                 batch_size: int,
                 log_level: int,
                 callbacks: list[Callback]=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_level = log_level
        self.callbacks = callbacks