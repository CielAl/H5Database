"""
    Implementation of the core Database class.
"""
import inspect
import os
# patch extraction and dataset split: Use 1 pair of train-validation to due to limitation of time
from PIL import Image as PilImage
import numpy as np
import tables
from tqdm import tqdm

from .abstract_database import AbstractDB
import glob
from h5database.common import Split
from typing import Dict, Tuple, Sequence, Callable, Any, Union
from lazy_property import LazyProperty
from h5database.common import get_path_limit
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


class Database(AbstractDB):

    def __init__(self, **kwargs):
        """

        Keyword Args:
            **kwargs: See the __init__ of h5database.database.abstract_database.AbstractDB
        """

        super().__init__(**kwargs)

    def generate_table_name(self, phase: str) -> Tuple[str, str]:
        """
        Override from AbstractDB. Define the output pytable name as database_name_phase.pytable
        Args:
            phase (str): Phase of dataset, e.g. train or val.

        Returns:
            pytable_full_path (str): Full file path of exported pytable given the phase.
            pytable_dir (str): The export dir of pytable
        """

        pytable_dir = os.path.join(self.export_dir)
        pytable_full_path = os.path.join(pytable_dir, "%s_%s%s" % (self.database_name, phase, '.pytable'))
        return pytable_full_path, pytable_dir

    def is_instantiated(self, phase):
        """
        Return false if no hdf5 found on the export path.
        Args:
            phase (str): Phase of the pytable, e.g. train.

        Returns:

        """

        file_path = self.generate_table_name(phase)[0]
        return os.path.exists(file_path)

    def initialize(self, force_overwrite=False):
        """
        Write files to the new database, if the database is not already created, unless specify the force_overwrite
        as True
        Args:
            force_overwrite (bool): Whether overwrite the pytables if they already exist. Default is False.

        Returns:

        """
        has_uninitiated_phase = any([not self.is_instantiated(phase) for phase in self.phases])
        if has_uninitiated_phase or force_overwrite:
            self._write_data()
        else:
            print("DB exists under the export dir")

    def _write_data(self):
        """
        Helper function of initialize.
        Returns:

        """
        filters = tables.Filters(complevel=self.comp_level)
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

    def __getitem__(self, phase_index_tuple: Union[Tuple[str, int], int]):
        """
        Enables subscription by phase and index: Database[phase, index]
        Args:
            phase_index_tuple (Tuple[str, int]): Tuple(phase, index). Specify the phase name and the item id.

        Returns:
            All types of data given the index.
        """
        logger.debug(type(phase_index_tuple))
        if isinstance(phase_index_tuple, Tuple):
            phase, index = phase_index_tuple
        else:
            phase = self.phases[0]
            index = phase_index_tuple
            logger.warning(f"integer index. Use 'train' as the default phase.")
        with tables.open_file(self.generate_table_name(phase)[0], 'r') as pytable:
            # image = getattr(pytable.root, self.types[0])[index, ]
            # label = getattr(pytable.root, self.types[1])[index, ]
            results = tuple(getattr(pytable.root, type_name)[index, ] for type_name in self.types)
        return results

    def size(self, phase) -> int:
        """
        Size of the pytable as number of rows (of h5array) given the phase.
        Args:
            phase (str): Phase of the pytable.

        Returns:
            H5Array Size as int.
        """
        with tables.open_file(self.generate_table_name(phase)[0], 'r') as pytable:
            return getattr(pytable.root, self.types[0]).shape[0]

    def __len__(self):
        """
        Length. Size of the datatable is defined as the sum of number of entries in all phases.
        Returns:
            Size of the database.
        """
        return sum([self.size(phase) for phase in self.phases])

    def peek(self, phase) -> Tuple:
        """
        Peek the shape of items given phase.
        Args:
            phase (str): Phase of pytable.

        Returns:
            Shape of data for each type given the phase.
        """
        table_name = self.generate_table_name(phase)[0]
        with tables.open_file(table_name, 'r') as pytable:
            return tuple(getattr(pytable.root, data_type).shape for data_type in self.types)

    def _validate_shape_key(self, data_shape: Dict[str, Tuple[int, ...]]):
        """
        Validate whether the types attribute aligns with the keys of the data_shape
        Args:
            data_shape (Dict[str, Tuple[int, ...]]): data_shape definition given by the constructor.

        Raises:
            AssertionError
        """
        assert (sorted(list(data_shape.keys())) == sorted(self.types))

    def get_files(self) -> Sequence[str]:
        """
        Default behavior getting the source files: get the list of files matching the pattern under the file_dir.
        Returns:
            files (Sequence[str]): Full paths of all selected source files.
        """
        file_pattern = os.path.join(self.file_dir, self.pattern)
        files = glob.glob(file_pattern)
        return files

    @LazyProperty
    def types(self) -> Sequence[str]:
        """
        Returns:
           self._types (str): Types of data specified in the shape_dict
        """
        if not hasattr(self, '_types') or self._types is None:
            self._types = self.parse_types(shape_dict=None)
        return self._types

    # override
    def parse_types(self, shape_dict: Dict[str, Tuple[int, ...]] = None) -> Sequence[str]:
        """
        Override from the superclass. Extract the name of data_types from the given shape_dict. \
        If shape_dict is not None, then the types are given by its keys. Otherwise use the \
        pre-defined train_name and val_name property.
        Args:
            shape_dict (Dict[str, Tuple[int, ...]]): Name-Value pair of <type, shape>. Default is None.
        Returns:
            type_names (Sequence[str]): Sequence of the names of types.
        """
        if shape_dict is not None:
            type_names = list(shape_dict.keys())
        else:
            phases = AbstractDB.train_name(), AbstractDB.val_name()
            type_names_list = []
            for phase in phases:
                with tables.open_file(self.generate_table_name(phase)[0], 'r') as pytable:
                    type_names_list.append(tuple(pytable.root.types[:]))
            type_agree: bool = len(set(type_names_list)) <= 1
            assert type_agree, f"type_names across phase disagree. " \
                f"Got: {type_names_list}"
            type_names = [x.decode('utf-8') for x in type_names_list[0]]
        return type_names

    def init_split(self, stratified_labels=None):
        """
        Initialize the data split and shuffle using K_fold. If labels are given, using stratified K-fold
        to preserve the label balance in each split.
        Args:
            stratified_labels (Sequence): A sequence of labels. Default is None.

        Returns:
            splits (Dict[str, np.ndarray]): File split (specified by file_id in the file_list) per phase.
        """
        splits = dict()
        splits[AbstractDB.train_name()], splits[AbstractDB.val_name()] = \
            next(Split.k_fold_split(self.num_split, shuffle=self.shuffle, file_list=self.file_list,
                                    stratified_labels=stratified_labels))
        return splits

    @staticmethod
    def _init_atoms(row_atom_func: Callable,
                    data_shape_dict: Dict[str, Tuple[int, ...]]) -> Dict[str, tables.Atom]:
        """
        Init the atoms for all types of data. Size of the atoms is defined by the shape.
        Args:
            row_atom_func (Callable): Constructor to create the
            data_shape_dict (Dict[str, Tuple[int, ...]])): Name-value pair of <type_name, shape>.
        Returns:
            atoms (Dict[str, tables.atom]): Dict of atoms per data type.
        """
        atoms = {}
        for k, v in data_shape_dict.items():
            atoms[k] = row_atom_func(shape=tuple(v))
        return atoms

    def refresh_atoms(self):
        """
        Recreate the self._atoms using _init_atoms.
        Returns:
            self._atoms (Dict[str, tables.atom]): Dict of atoms per data type.
        """
        self._atoms = type(self)._init_atoms(self.row_atom_func, self.data_shape)
        return self._atoms

    @staticmethod
    def prepare_export_directory(pytable_dir: str):
        """
        Makedir if the pytable_dir does not exist.
        Args:
            pytable_dir (str): Target directory.

        Returns:

        """
        if not os.path.exists(pytable_dir):
            os.makedirs(pytable_dir)

    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...


class DbHelper(object):
    """
    Parent class for all helper classes.
    """
    def __init__(self, database: AbstractDB):
        """
        Args:
            database (AbstractDB): Associate the helper class with the target database.
        """
        self._database = database

    @property
    def database(self) -> AbstractDB:
        return self._database


class TaskManager(DbHelper):
    """
        Pipeline control of all tasks: data extraction/writing, weight collecting, H5Array creation, and etc.
    """
    def __init__(self, database: AbstractDB):
        """
        Initialize the atoms for meta-data (types, valid tag, and splits)
        Args:
            database (AbstractDB): Associated Database object
        """
        super().__init__(database)
        self.filename_atom = tables.StringAtom(itemsize=255)
        self.types_atom = tables.StringAtom(itemsize=255)
        # whether the patch is valid.
        self.valid_atom = tables.BoolAtom(shape=(), dflt=False)
        # save the meta info: split
        # noinspection PyArgumentList
        self.file_list_atom = tables.StringAtom(itemsize=get_path_limit())
        # noinspection PyArgumentList
        self.split_atom = tables.IntAtom(shape=(), dflt=False)

        self.hdf5_organizer = H5Organizer(self.database, self.database.group_level)
        self.data_extractor = DataExtractor(self.database)
        self.weight_writer = WeightCollector(self.database, self.data_extractor,
                                             weight_counter=self.database.weight_counter_callable)
        self.data_size = {}

    @property
    def types(self):
        return self.database.types

    def write(self, phase, filters):
        """
        Write data to the H5array by phase.
        Args:
            phase (str): Phase of the pytable.
            filters (tables.Filter): Filters for H5array. Compression-level specified.

        Returns:

        """
        self.hdf5_organizer.build_patch_array(phase, 'filename', self.filename_atom)
        self.hdf5_organizer.build_patch_array(phase, 'file_list', self.file_list_atom)
        self.hdf5_organizer.build_patch_array(phase, 'valid', self.valid_atom)
        self.hdf5_organizer.build_patch_array(phase, 'split', self.split_atom, group_level=H5Organizer.LEVEL_PATCH)
        self.hdf5_organizer.build_patch_array(phase, 'types', self.types_atom, group_level=H5Organizer.LEVEL_PATCH)
        with self.hdf5_organizer.pytables[phase]:
            self.database.refresh_atoms()
            self._create_h5array_by_types(phase, filters)
            self._fetch_all_files(phase)
            self.weight_writer.write_class_weight_to_db(self.hdf5_organizer, phase)
            # self.database.splits
            self.hdf5_organizer.h5arrays[phase]['split'].append(self.database.splits[phase])
            self.hdf5_organizer.h5arrays[phase]['types'].append(self.types)
            self.hdf5_organizer.h5arrays[phase]['file_list'].append(self.database.file_list)

    def _create_h5array_by_types(self, phase, filters):
        """
        Create H5arrays given phase.
        Args:
            phase (str): Phase of pytable.
            filters (tables.Filter): Filters that specified the compression-level.

        Returns:

        """
        for category_type in self.types:
            self.hdf5_organizer.build_patch_array(phase, category_type, self.database.atoms[category_type],
                                                  filters)

    def __fetch_data(self, file: str):
        """
            Extract patches (images, masks, or labels), valid_tag, and extra_info
            Helper function of _fetch_all_files
        Args:
            file (str): full path of a source file.

        Returns:
            patches (Dict[str, Tuple]): Name-value pair of <type_name, data>. All types of data are aggregated.
            valid_tag (): Sequence of bool/int that specify whether a row is valid/qualified.
            extra_info (Any): Any extra information to report.
        """
        patches_group, valid_tag, type_order, extra_info = self.data_extractor.extract(file)
        patches: Dict[str, Tuple] = {type_name: data for (type_name, data) in zip(type_order, patches_group)}
        return patches, valid_tag, extra_info

    def _fetch_all_files(self, phase):
        """
        Inner iteration to extract data from files.
        Args:
            phase ():

        Returns:

        """
        for idx, file_id in enumerate(tqdm(self.database.splits[phase])):
            file = self.database.file_list[file_id]
            patches, valid_tag, extra_info = self.__fetch_data(file)
            valid_tag = np.atleast_1d(valid_tag)
            if any(list(valid_tag)) or self.database.write_invalid:
                self.hdf5_organizer.write_data_to_array(phase, patches, self.data_size)
                self.weight_writer.weight_accumulate(file, self.types, patches, extra_info)
                self.hdf5_organizer.write_file_names_to_array(phase, file, patches, category_type="filename")
                self.hdf5_organizer.h5arrays[phase]['valid'].append(valid_tag)

                ...
            else:
                # do nothing. leave it blank here for: (1) test. (2) future works
                ...

    def __enter__(self):
        """
        Entry of with statement to write data.
        Returns:

        """
        return self

    def __exit__(self, type_name, value, traceback):
        """
        Flush the h5array upon exit.
        Args:
            type_name ():
            value ():
            traceback ():

        Returns:

        """
        self.flush()

    def flush(self):
        """
        Flush the h5arrays of all phases.
        Returns:

        """
        for phase in self.hdf5_organizer.phases:
            self.hdf5_organizer.flush(phase)


class DataExtractor(DbHelper):
    """
    Controller of Data Extration workflow.
    """
    def __init__(self, database: AbstractDB):
        super().__init__(database)
        self._extract_callable = database.extract_callable
        assert not inspect.isclass(self._extract_callable), "Expect an instantiated callable. " \
                                                            "A class variable is encountered."
        self.meta = type(self)._default_meta(self.database.meta)
        self._meta_load_database(self.database)

    def _meta_load_database(self, database: AbstractDB):
        """
        Associate the target database fields to the "meta" field that is passed to the Extractor.
        Args:
            database (AbstractDB): Associated database.
        Returns:

        """
        assert hasattr(database, 'data_shape'), "data_shape uninitialized"
        self.meta.update({'data_shape': database.data_shape})

    @property
    def extract_callable(self):
        return self._extract_callable

    # part of common default values
    @classmethod
    def _default_meta(cls, meta):
        """
        Fill-in the meta.
        Args:
            meta ():

        Returns:

        """
        meta['stride_size'] = meta.get('stride_size', 128)
        meta['tissue_area_thresh'] = meta.get('tissue_area_thresh', 0.95)
        meta['interp'] = meta.get('interp', PilImage.NONE)
        meta['resize'] = meta.get('resize', 1)
        # meta = SimpleNamespace(**meta)
        return meta

    def extract(self, file: str) -> Tuple[Tuple[Any, ...], Sequence[bool], Sequence[str], Any]:
        """
        Extract data, e.g. patches from the source file.
        Args:
            file (str): Full file path.

        Returns:
            Tuple[object, ...] as the group of data extracted from the file. Length of the object
            must be the same.
            Sequence[bool] as a sequence of validation tag, identifying whether the corresponding
            object is qualified.
            Sequence[str]
            object as
        """
        return self.extract_callable(self, file)


class WeightCollector(DbHelper):
    """
        Controller.
    """
    def __init__(self, database: AbstractDB, extractor: DataExtractor, **kwargs):
        """

        Args:
            database (AbstractDB):  Associated Database.
            extractor (DataExtractor):  Associated Extractor Callable
            **kwargs ():
        """
        super().__init__(database)
        self.weight_counter_callable = kwargs.get('weight_counter', self.database.weight_counter_callable)
        self._totals = self._new_weight_storage()
        self._extractor = extractor

    @property
    def extractor(self):
        return self._extractor

    @property
    def totals(self):
        """

        Returns:
            The cached class counts.
        """
        return self._totals

    def is_count_weight(self):
        """
        Validate Weight Collection should be performed.
        Returns:
            True only if the "enable_weight" field in the associated database obj is true, and class \
            names are defined.
        """
        return self.database.enable_weight and self.database.classes is not None

    def weight_accumulate(self, file: str,
                          type_names: Sequence[str],
                          patch_group: Dict[str, object],
                          extra_information: Any):
        """
        A wrapper to accumulate class count, if is_count_weight is validated, given the
        callable that defines the behavior of weight accumulation.
        Args:
            file (): Filename.
            type_names (): Type names defined in the dataset.
            patch_group (): Extracted data of all types from the source file.
            extra_information (): Any extra parameters to pass.

        Returns:

        """
        if self.is_count_weight():
            self.weight_counter_callable(self,
                                         file,
                                         type_names,
                                         patch_group,
                                         extra_information)

    def _new_weight_storage(self):
        if self.is_count_weight():
            totals = np.zeros(len(self.database.classes))
        else:
            totals = None
        return totals

    def write_class_weight_to_db(self, hdf5_organizer, phase, renew_when_done=True):
        if self.is_count_weight():
            n_pixels = hdf5_organizer.build_meta_data(phase, 'class_sizes', self.totals)
            n_pixels[:] = self.totals
            if renew_when_done:
                self._totals = self._new_weight_storage()


class H5Organizer(DbHelper):
    LEVEL_PATCH = 0
    LEVEL_GROUP = 1

    def __init__(self, database: AbstractDB, group_level: int):
        """

        Args:
            database (): Associated Database instance.
            group_level (): 0 if no grouping. 1 if group data (e.g. patches) extracted from one file
                            in a row of a VLarray. Each row of a VLarray is a list of data.
        """
        super().__init__(database)
        self._h5arrays = type(self)._empty_h5arrays_dict(self.phases, self.types)
        self._pytables = self._new_pytables_from_database(self.phases)
        self.group_level = group_level

    def flush(self, phase):
        for type_name, h5array in self._h5arrays[phase].items():
            self._h5arrays[phase][type_name] = None
        self._h5arrays[phase] = dict()
        self.pytables[phase].close()

    @property
    def h5arrays(self):
        return self._h5arrays

    @property
    def pytables(self):
        return self._pytables

    @property
    def atoms(self):
        return self.database.atoms

    @property
    def phases(self):
        return self.database.phases

    @property
    def types(self):
        return self.database.types

    @staticmethod
    def _get_h5_shapes(chunk_width: int):
        """
        Helper function to define the shape of the h5array.
        Args:
            chunk_width ():

        Returns:

        """
        chunk_shape = [chunk_width]
        h5_shape = [0]
        return h5_shape, chunk_shape

    '''Precondition:h5arrays and pytables initialized/created
    '''

    def build_patch_array(self, phase: str,
                          category_type: str,
                          atom: tables.Atom,
                          filters: tables.Filters = None,
                          expectedrows: int = None,
                          group_level: int = None) -> Dict:
        """
        Create H5array for extracted data.
        Args:
            phase (): Phase Key.
            category_type (): Type name.
            atom (): Type of the atom for the array.
            filters (): Filter instance for the H5array
            expectedrows (): User Estimation of number of rows of the H5array.
            group_level (): Whether data are grouped. If 0: no grouping. \
            If 1: group by source file, using VLarray.
        Returns:
            self.h5arrays
        """
        h5_shape, chunk_shape = type(self)._get_h5_shapes(self.database.chunk_width)

        # **kwargs for the create_func
        params = {
            'atom': atom,
            'filters': filters,
            'chunkshape': chunk_shape,
        }
        if expectedrows is not None and isinstance(expectedrows, int):
            params['expectedrows'] = expectedrows
        if group_level is None:
            group_level = self.group_level
        if group_level is type(self).LEVEL_PATCH:
            create_func = self.pytables[phase].create_earray
            params['shape'] = h5_shape
        elif group_level is type(self).LEVEL_GROUP:
            create_func = self.pytables[phase].create_vlarray
        else:
            raise ValueError('Invalid Group Level:' + str(self.group_level))
        self.h5arrays[phase][category_type] = create_func(self.pytables[phase].root, category_type, **params)
        return self.h5arrays[phase][category_type]

    # carrays. metadata
    def build_meta_data(self, phase: str, category_type: str, template_data: Any):
        """
        Create H5Array for metadata, which is not tied to extracted data array.
        Args:
            phase ():
            category_type ():
            template_data ():

        Returns:

        """
        self.h5arrays[phase][category_type] = self.pytables[phase]\
                                                  .create_carray(self.pytables[phase].root,
                                                                 category_type,
                                                                 tables.Atom.from_dtype(template_data.dtype),
                                                                 template_data.shape)
        return self.h5arrays[phase][category_type]

    # todo
    def write_data_to_array(self, phase, patches_dict, data_size=None):
        for type_name in self.types:
            logger.debug(patches_dict[type_name].shape)
            self.h5arrays[phase][type_name].append(patches_dict[type_name])
            # write size of data into h5 - todo
            if data_size is not None:
                data_size[type_name] = data_size.get(type_name, 0) + np.asarray(patches_dict[type_name]).shape[0]

    # todo
    def write_file_names_to_array(self, phase: str,
                                  file: str,
                                  patches: Dict[str, Any],
                                  category_type: str = "filename"):
        """
        Write the source filename of a group of corresponding extracted
        data to an h5array, given the phase and type.
        Perform write-in if only patches is not None, and each type name of "patches"
        is defined in the database.
        Args:
            phase (): Phase key.
            file ():  Full filename
            patches (): Extracted data.
            category_type (): Type name of this field. Default "filename"

        Returns:

        """
        # (patches[self.types[0]] is not None) and (patches[self.types[1]] is not None)
        if patches and all([patches[type_name] is not None for type_name in self.types]):
            # [file for x in range(patches[self.types[0]].shape[0])]
            filename_list = [file] * patches[self.types[0]].shape[0]
            if filename_list:
                self.h5arrays[phase][category_type].append(filename_list)
            else:
                ...  # do nothing. leave it here for tests - todo

    @staticmethod
    def _empty_h5arrays_dict(phases: Sequence[str], types: Sequence[str]) -> Dict[str, Dict]:
        """
        Init the new empty dictionary to hold h5arrays by phase and data type.
        Args:
            phases (): phase key.
            types ():  Data type name, e.g. "img" or "mask".

        Returns:

        """
        h5arrays: Dict[str, Dict] = dict()
        for phase in phases:
            h5arrays[phase]: Dict = dict()
            for type_name in types:
                h5arrays[phase][type_name] = None
        return h5arrays

    def _new_pytables_from_database(self, phases) -> Dict[str, tables.file.File]:
        """
        Initialize the file handle in the Pytable Dictionary.
        Args:
            phases (): phase key, e.g. 'train' or 'val'

        Returns:
            Dict[str, tables.file.File]. Dict of pytable handle by phase.
        """
        pytables: Dict[str, tables.file.File] = dict()
        for phase in phases:
            pytable_full_path, pytable_dir = self.database.generate_table_name(phase)
            type(self.database).prepare_export_directory(pytable_dir)
            pytables[phase] = tables.open_file(pytable_full_path, mode='w')
        return pytables
