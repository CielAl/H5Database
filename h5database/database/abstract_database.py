from abc import ABC, abstractmethod
from lazy_property import LazyProperty
from typing import Tuple, Dict, Callable, Sequence
import tables


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
        """

        Keyword Args:
             export_dir (str): The output directory for pytables.
             database_name (str): The name of database. Also defined the file name of the exported pytables.
             readonly (bool): Whether the instance is only to read pytables or not. Default False.
             file_dir (str): Directory of source files (to be extracted into H5database).
             data_shape (Dict[str, Tuple[int, ...]]): The atom shape of each type of output (e.g. image or label).
             group_level (int): 0 if no grouping. 1 if group patches in a file into a VL_Array.
             extract_callable (Callable): The callable to extract data from source files.
             write_invalid (bool): True if retain invalid extracted data in the h5array. Default: False.
             chunk_width (int): Size of chunk for Pytable.
             num_split (int): Number of splits if using k-fold split strategy. Default 10.
             shuffle (bool): True if shuffle the order of source files. Default True.
             pattern (str): pattern of source files in the source directory. Default: '*.png'
             weight_counter_callable (Callable): The callable to collect class weights. Default None.
             enable_weight (bool): True if collect the class weight. Default False.
             classes (Sequence[str]): A sequence of class names. Default None.
             file_list (Sequence[str]): The source file list.
                                        Default is all files matching the pattern under the file_dir
             splits (Dict[str, Sequence[int]]): Splits of file corresponding to phases (train/validation etc.).
                                                Default is the k-fold split.
             meta (Dict): Extra parameters to be passed to callables (extract_callable or weight_counter_callable)
             row_atom_func (Callable): The constructor/builder of the atom of the row of EArray/VLArray.
                                        Default is tables.UInt8Atom
             comp_level (int): Compression Level for the filter of pytable. Default value is 3.

        """
        self.KEY_TRAIN: str = type(self).train_name()
        self.KEY_VAL: str = type(self).val_name()
        self.export_dir: str = kwargs['export_dir']
        self.database_name: str = kwargs['database_name']
        self.readonly: bool = kwargs.get('readonly', False)
        # for style only. Initialized by the setter below.
        self._atoms = None
        if not self.readonly:
            self.__init_write_params(**kwargs)
        else:
            self._types = self.parse_types()

    def __init_write_params(self, **kwargs):
        """

        Args:
            **kwargs (): See the docstring of __init__
        Returns:

        """
        self.file_dir: str = kwargs['file_dir']
        self.data_shape: Dict[str, Tuple[int, ...]] = kwargs['data_shape']
        self._types = self.parse_types(shape_dict=self.data_shape)
        self._validate_shape_key(self.data_shape)
        self.group_level: int = kwargs['group_level']
        self.extract_callable = kwargs['extractor']
        # refactor later
        self.write_invalid: bool = kwargs.get('write_invalid', False)
        self.chunk_width: int = kwargs.get('chunk_width', 4)
        self.num_split: int = kwargs.get('num_fold', 10)
        self.shuffle: bool = kwargs.get('shuffle', True)
        self.pattern: str = kwargs.get('pattern', '*.png')

        self.weight_counter_callable = kwargs.get('weight_counter', None)
        self.enable_weight: bool = kwargs.get('class_weight', False)
        self.classes: Sequence[str] = kwargs.get('class_names', None)
        self.file_list: Sequence[str] = kwargs.get('file_list', self.get_files())
        self.splits: Dict[str, Sequence[int]] = kwargs.get('split', self.init_split())
        # todo coordinate can be inserted into data_shape, and implemented by Extractor
        # for Database itself, meta is not handled until passed to DataaExtractor
        self.meta: Dict = kwargs.get('meta', {})
        self.row_atom_func: Callable = kwargs.get('row_atom_func', tables.UInt8Atom)
        self.comp_level: int = kwargs.get('comp_level', 3)
        self.refresh_atoms()

    @abstractmethod
    def __getitem__(self, index):
        """
        Read the pytable by index subscription.
        Args:
            index ():

        Returns:

        """
        ...

    @abstractmethod
    def generate_table_name(self, param):
        """
            Define the filename of the pytables
        Args:
            param ():

        Returns:

        """
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
        """

        Returns:

        """
        return self.splits.keys()

    @LazyProperty
    @abstractmethod
    def types(self):
        ...

    @abstractmethod
    def _validate_shape_key(self, data_shape):
        ...

    @abstractmethod
    def get_files(self) -> Sequence[str]:
        """
        Behavior to extract the source file list.
        Returns:

        """
        ...

    @abstractmethod
    def parse_types(self, shape_dict: Dict[str, Tuple[int, ...]] = None):
        """
        Extract names of data types from the dict of shapes. Data types as the purpose of data,
        i.e. img, mask, or labels etc.
        Args:
            shape_dict (Dict[str, Tuple[int, ...]]): The shape_dict per type.

        Returns:

        """
        ...

    @abstractmethod
    def init_split(self, stratified_labels=None):
        ...

    @abstractmethod
    def refresh_atoms(self):
        """
        Recreate the row atoms every time an entry is fetched into the h5array.
        Otherwise exception may occur.
        Returns:

        """
        ...

    @staticmethod
    @abstractmethod
    def prepare_export_directory(pytable_dir: str):
        """
        If output directory does not exist, then create it.
        Args:
            pytable_dir (str): The expected output directory

        Returns:

        """
        ...

    @abstractmethod
    def __enter__(self):
        ...

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        ...
