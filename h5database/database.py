from h5database.util import TaskDispatcher
import glob
import os
# patch extraction and dataset split: Use 1 pair of train-validation to due to limitation of time

from typing import List, Tuple, Dict, Callable, Iterable
import tables
from lazy_property import LazyProperty
from .split import Split

_TRAIN_NAME = 'train'
_VAL_NAME = 'val'


@property
def train_name():
    return _TRAIN_NAME

@property
def val_name():
    return _VAL_NAME


class Database(object):

    def __init__(self, **kwargs):
        """
        Constructor
        :param kwargs:
        """
        self.KEY_TRAIN: str = _TRAIN_NAME
        self.KEY_VAL: str = _VAL_NAME
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
            self.patch_pair_extractor_func: Callable = kwargs['extractor']
            # refactor later
            self.write_invalid: bool = kwargs.get('write_invalid', False)
            self.chunk_width: int = kwargs.get('chunk_width', 1)
            self.num_split: int = kwargs.get('num_fold', 10)
            self.shuffle: bool = kwargs.get('shuffle', True)
            self.pattern: str = kwargs.get('pattern', '*.png')

            self.weight_counter_func: Callable = kwargs.get('weight_counter', None)
            self.enable_weight: bool = kwargs.get('class_weight', False)
            self.classes: Iterable[str] = kwargs.get('class_names', None)

            self.file_list: Iterable[str] = kwargs.get('file_list', self.get_files())
            self.splits: Dict[str, Iterable[int]] = kwargs.get('split', self.init_default_split())

            # for Database itself, meta is not handled until passed to DataaExtractor
            self.meta: Dict = kwargs.get('meta', {})

            self.filenameAtom = tables.StringAtom(itemsize=255)
            # whether the patch is valid.
            self.validAtom = tables.BoolAtom(shape=(), dflt=False)

            # save the meta info: split
            # noinspection PyArgumentList
            self.splitAtom = tables.IntAtom(shape=(), dflt=False)
            self.row_atom_func: Callable = kwargs.get('row_atom_func', tables.UInt8Atom)
            self.refresh_atoms()

    def _validate_shape_key(self, data_shape):
        assert (sorted(list(data_shape.keys())) == sorted(self.types))

    @staticmethod
    def _init_atoms(row_atom_func, data_shape_dict):
        atoms = {}
        for k, v in data_shape_dict.items():
            atoms[k] = row_atom_func(shape=tuple(v))
        return atoms

    def refresh_atoms(self):
        self._atoms = type(self)._init_atoms(self.row_atom_func, self.data_shape)
        return self._atoms



    @staticmethod
    def parse_types(shape_dict: Dict[str, Tuple[int, ...]]):
        return list(shape_dict.keys())

    @LazyProperty
    def types(self):
        return self._types

    @property
    def atoms(self):
        return self._atoms

    @property
    def phases(self):
        return self.splits.keys()

    '''
        Get the list of files by the pattern and location, if not specified by user.
    '''

    def get_files(self):
        file_pattern = os.path.join(self.file_dir, self.pattern)
        files = glob.glob(file_pattern)
        return files

    '''
        Initialize the data split and shuffle.
    '''
    def init_default_split(self, stratified_labels=None):
        splits = dict()
        splits[_TRAIN_NAME], splits[_VAL_NAME] = next(Split.k_fold_database(self, stratified_labels=stratified_labels))
        return splits

    '''
        Read the file and return (img,label,success,meta).
        Invoke the patch_pair_extractor_func, which is a function handle. So "self" must be explicitly passed to 
        the inputs.
    '''

    def _img_label_patches(self, file):
        # patch_pair_extractor_func is a callback: manually pass "self" as its 1st arg
        return self.patch_pair_extractor_func(self, file)

    '''
        Generate the Table name from database name.
    '''

    def generate_table_name(self, phase):
        pytable_dir = os.path.join(self.export_dir)
        pytable_full_path = os.path.join(pytable_dir, "%s_%s%s" % (self.database_name, phase, '.pytable'))
        return pytable_full_path, pytable_dir

    '''
        Return false if no hdf5 found on the export path.
    '''

    def is_instantiated(self, phase):
        file_path = self.generate_table_name(phase)[0]
        return os.path.exists(file_path)

    '''
        non-overridable
    '''

    def initialize(self, force_overwrite=False):
        if (not self.is_instantiated(_TRAIN_NAME)) or (not self.is_instantiated(_VAL_NAME)) or force_overwrite:
            self._write_data()

    def _write_data(self):
        filters = tables.Filters(complevel=5)
        with TaskDispatcher(self) as self.task_dispatcher:
            for phase in self.phases:
                self.task_dispatcher.write(phase, filters)
        return self.task_dispatcher.data_size

    '''
        Slice the chunk of patches out of the database.
        Precondition: Existence of pytable file. Does not require to call "_write_data" as long as pytables exist
        Args:
            phase_index_tuple: ('train/val',index)
        Returns:
            images, labels(ground truth)
    '''

    def __getitem__(self, phase_index_tuple):
        phase, index = phase_index_tuple
        with tables.open_file(self.generate_table_name(phase)[0], 'r') as pytable:
            image = getattr(pytable.root, self.types[0])[index, ]
            label = getattr(pytable.root, self.types[1])[index, ]
        return image, label

    def size(self, phase):
        with tables.open_file(self.generate_table_name(phase)[0], 'r') as pytable:
            return getattr(pytable.root, self.types[0]).shape[0]

    def __len__(self):
        return sum([self.size(phase) for phase in self.phases])

    def peek(self, phase):
        with tables.open_file(self.generate_table_name(phase)[0], 'r') as pytable:
            return (getattr(pytable.root, self.types[0]).shape,
                    getattr(pytable.root, self.types[1]).shape)

    @staticmethod
    def prepare_export_directory(pytable_dir):
        if not os.path.exists(pytable_dir):
            os.makedirs(pytable_dir)
