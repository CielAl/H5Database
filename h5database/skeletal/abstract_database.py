from abc import ABC, abstractmethod
from lazy_property import LazyProperty
from typing import List, Tuple, Dict, Callable, Sequence
import tables
from h5database.skeletal import ExtractCallable
from h5database.skeletal import WeightCounterCallable


class AbstractDB(ABC):
    _TRAIN_NAME = 'train'
    _VAL_NAME = 'val'

    @classmethod
    def train_name(cls) -> str:
        return cls._TRAIN_NAME

    @classmethod
    def val_name(cls) -> str:
        return cls._VAL_NAME

    def __init__(self, **kwargs):
        self.KEY_TRAIN: str = type(self).train_name()
        self.KEY_VAL: str = type(self).val_name()
        self.export_dir: str = kwargs['export_dir']
        self.database_name: str = kwargs['database_name']
        self.readonly: bool = kwargs.get('readonly', False)
        # for style only. Initialized by the setter below.
        self._atoms = None

        if not self.readonly:
            self.file_dir: str = kwargs['file_dir']
            self.data_shape: Dict[str, Tuple[int, ...]] = kwargs['data_shape']
            self._types: List[str] = type(self).parse_types(self.data_shape)

            self._validate_shape_key(self.data_shape)
            self.group_level: int = kwargs['group_level']
            self.extract_callable: ExtractCallable = kwargs['extractor']
            # refactor later
            self.write_invalid: bool = kwargs.get('write_invalid', False)
            self.chunk_width: int = kwargs.get('chunk_width', 1)
            self.num_split: int = kwargs.get('num_fold', 10)
            self.shuffle: bool = kwargs.get('shuffle', True)
            self.pattern: str = kwargs.get('pattern', '*.png')

            self.weight_counter_callable: WeightCounterCallable = kwargs.get('weight_counter', None)
            self.enable_weight: bool = kwargs.get('class_weight', False)
            self.classes: Sequence[str] = kwargs.get('class_names', None)

            self.file_list: Sequence[str] = kwargs.get('file_list', self.get_files())
            self.splits: Dict[str, Sequence[int]] = kwargs.get('split', self.init_split())

            # for Database itself, meta is not handled until passed to DataaExtractor
            self.meta: Dict = kwargs.get('meta', {})

            self.row_atom_func: Callable = kwargs.get('row_atom_func', tables.UInt8Atom)
            self.refresh_atoms()

    @abstractmethod
    def __getitem__(self, index):
        ...

    @abstractmethod
    def generate_table_name(self, param):
        ...

    @abstractmethod
    def initialize(self, **kwargs):
        ...

    @abstractmethod
    def __len__(self):
        ...

    @property
    def atoms(self):
        return self._atoms

    @property
    def phases(self):
        return self.splits.keys()

    @LazyProperty
    def types(self):
        return self._types

    @abstractmethod
    def _validate_shape_key(self, data_shape):
        ...

    '''
        Get the list of files by the pattern and location, if not specified by user.
    '''
    @abstractmethod
    def get_files(self):
        ...

    @staticmethod
    @abstractmethod
    def parse_types(shape_dict: Dict[str, Tuple[int, ...]]):
        ...

    @abstractmethod
    def init_split(self, stratified_labels=None):
        ...

    @abstractmethod
    def refresh_atoms(self):
        ...

    @staticmethod
    @abstractmethod
    def prepare_export_directory(pytable_dir):
        ...
