from abc import ABC, abstractmethod
from typing import Sequence, Dict
import numpy as np
from h5database.database.helper import WeightCollector


class WeightCounterCallable(ABC):

    @classmethod
    def __call__(cls, collector: WeightCollector, file: str, type_names: Sequence[str],
                 patch_group: Dict, extra_info) -> np.ndarray:
        cls.__count(collector, file, type_names, patch_group, extra_info)
        return collector.totals

    @staticmethod
    @abstractmethod
    def __count(collector: WeightCollector, file: str, type_names: Sequence[str],
                patch_group: Dict, extra_info: object):
        ...
