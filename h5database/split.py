from sklearn import model_selection
from typing import Iterable
from .database import Database


class Split:
    def __init__(self):
        ...

    @staticmethod
    def k_fold_split(num_split: int, shuffle: bool, file_list: Iterable[str], stratified_labels: Iterable = None):
        if stratified_labels is None:
            return iter(model_selection.KFold(n_splits=num_split, shuffle=shuffle).split(file_list))
        return iter(model_selection.StratifiedKFold(n_splits=num_split, shuffle=shuffle)
                    .split(file_list, stratified_labels))

    @classmethod
    def k_fold_database(cls, database: Database, stratified_labels: Iterable = None):
        return cls.k_fold_split(database.num_split, database.shuffle, database.file_list, stratified_labels)
