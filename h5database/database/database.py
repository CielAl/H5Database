import os
# patch extraction and dataset split: Use 1 pair of train-validation to due to limitation of time
import tables
from h5database.skeletal import AbstractDB
from h5database.database.helper import TaskManager
import glob
from h5database.common import Split
from typing import Dict, Tuple, Sequence
from lazy_property import LazyProperty


class Database(AbstractDB):

    def __init__(self, **kwargs):
        """
        Constructor
        :param kwargs:
        """
        super().__init__(**kwargs)

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
        if (not self.is_instantiated(type(self).train_name())) or (not self.is_instantiated(type(self).val_name())) \
                or force_overwrite:
            self._write_data()

    def _write_data(self):
        filters = tables.Filters(complevel=5)
        with TaskManager(self) as self.task_dispatcher:
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

    def _validate_shape_key(self, data_shape):
        assert (sorted(list(data_shape.keys())) == sorted(self.types))

    '''
        Get the list of files by the pattern and location, if not specified by user.
    '''
    def get_files(self):
        file_pattern = os.path.join(self.file_dir, self.pattern)
        files = glob.glob(file_pattern)
        return files

    @LazyProperty
    def types(self):
        if not hasattr(self, '_types') or self._types is None:
            self._types = self.parse_types(shape_dict=None)
        return self._types

    # override
    def parse_types(self, shape_dict: Dict[str, Tuple[int, ...]] = None) -> Sequence[str]:
        if shape_dict is not None:
            type_names = list(shape_dict.keys())
        else:
            phases = type(self).train_name(), type(self).val_name()
            type_names_list = []
            for phase in phases:
                with tables.open_file(self.generate_table_name(phase)[0], 'r') as pytable:
                    type_names_list.append(tuple(pytable.root.types[:]))
            type_agree: bool = len(set(type_names_list)) <= 1
            assert type_agree, f"type_names across phase disagree. " \
                f"Got: {type_names_list}"
            type_names = [x.decode('utf-8') for x in type_names_list[0]]
        return type_names

    '''
        Initialize the data split and shuffle.
    '''
    def init_split(self, stratified_labels=None):
        splits = dict()
        splits[type(self).train_name()], splits[type(self).val_name()] = \
            next(Split.k_fold_split(self.num_split, shuffle=self.shuffle, file_list=self.file_list,
                                    stratified_labels=stratified_labels))
        return splits

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
    def prepare_export_directory(pytable_dir):
        if not os.path.exists(pytable_dir):
            os.makedirs(pytable_dir)

    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...