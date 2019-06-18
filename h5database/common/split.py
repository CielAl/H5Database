from sklearn import model_selection
from typing import Iterable


class Split:

    @staticmethod
    def k_fold_split(num_split: int, shuffle: bool = False, file_list: Iterable[str] = None,
                     stratified_labels: Iterable = None):
        if stratified_labels is None:
            return iter(model_selection.KFold(n_splits=num_split, shuffle=shuffle).split(file_list))
        return iter(model_selection.StratifiedKFold(n_splits=num_split, shuffle=shuffle)
                    .split(file_list, stratified_labels))
