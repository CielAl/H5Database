import os
import re
from h5database.skeletal import WeightCounterCallable
from typing import Sequence, Dict
from h5database.database.database import WeightCollector


class WeightFile(WeightCounterCallable):
    """
        Accumulate by file counts.
    """
    # override
    @staticmethod
    def _count(collector: WeightCollector, file: str, type_names: Sequence[str], patch_group: Dict,
               extra_info: object):
        label_key: str = collector.extractor.extract_callable.label_key()
        label = patch_group[label_key]
        basename = os.path.basename(file)
        class_list = [idx for idx in range(len(collector.database.classes)) if
                      re.search(str(collector.database.classes[idx]), basename, re.IGNORECASE)]
        class_id = class_list[0]
        collector.totals[class_id] += len(label)


class WeightMaskPixelCallable(WeightCounterCallable):
    """
        Accumulate weight by mask pixels.
    """
    # override
    @staticmethod
    def _count(collector: WeightCollector, file: str, type_names: Sequence[str], patch_group: Dict,
               extra_info: object):
        label_key: str = collector.extractor.extract_callable.label_key()
        label = patch_group[label_key]
        for i, key in enumerate(collector.database.classes):
            collector.totals[1, i] += sum(sum(label[:, :, 0] == key))
