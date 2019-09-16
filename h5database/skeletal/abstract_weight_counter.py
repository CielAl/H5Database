from abc import ABC, abstractmethod
from typing import Sequence, Dict, Callable
import numpy as np
from h5database.database.database import WeightCollector


class WeightCounterCallable(ABC, Callable):
    """
        Callable Wrapper for Weight Counter.
    """
    @classmethod
    def __call__(cls, collector: WeightCollector, file: str, type_names: Sequence[str],
                 patch_group: Dict, extra_info) -> np.ndarray:
        """
            Callable Interface. "_count" is invoked internally.
        Args:
            collector (): Associated WeightCollector
            file ():    File Path.
            type_names (): Name of data types.
            patch_group (): Group of data array of each type.
            extra_info ():  Any extra_info obtained from DataExtractor.

        Returns:
            collector.totals: The cached np.ndarray that stores the class instance count.
        """
        cls._count(collector, file, type_names, patch_group, extra_info)
        return collector.totals

    @staticmethod
    @abstractmethod
    def _count(collector: WeightCollector, file: str, type_names: Sequence[str], patch_group: Dict,
               extra_info: object):
        """
        Helper function of the __call__. Behavior of weight accumulation.
        Args:
            collector (): Associated WeightCollector
            file ():    File Path.
            type_names (): Name of data types.
            patch_group (): Group of data array of each type.
            extra_info ():  Any extra_info obtained from DataExtractor.

        Returns:

        """
        ...
