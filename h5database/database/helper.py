from tqdm import tqdm
import numpy as np
import PIL
import tables
from typing import Dict, Tuple, Sequence
from h5database.skeletal import AbstractDB
import inspect


class DbHelper(object):
    def __init__(self, database: AbstractDB):
        self._database = database

    @property
    def database(self) -> AbstractDB:
        return self._database


class TaskManager(DbHelper):
    def __init__(self, database: AbstractDB):
        super().__init__(database)
        self.filename_atom = tables.StringAtom(itemsize=255)
        self.types_atom = tables.StringAtom(itemsize=255)
        # whether the patch is valid.
        self.valid_atom = tables.BoolAtom(shape=(), dflt=False)
        # save the meta info: split
        # noinspection PyArgumentList
        self.split_atom = tables.IntAtom(shape=(), dflt=False)  # todo

        self.hdf5_organizer = H5Organizer(self.database, self.database.group_level)
        self.data_extractor = DataExtractor(self.database)
        self.weight_writer = WeightCollector(self.database, self.data_extractor,
                                             weight_counter=self.database.weight_counter_callable)
        self.data_size = {}

    @property
    def types(self):
        return self.database.types

    def write(self, phase, filters):
        self.hdf5_organizer.build_patch_array(phase, 'filename', self.filename_atom)
        self.hdf5_organizer.build_patch_array(phase, 'valid', self.valid_atom)
        self.hdf5_organizer.build_patch_array(phase, 'split', self.split_atom, group_level=H5Organizer.LEVEL_PATCH)
        self.hdf5_organizer.build_patch_array(phase, 'types', self.types_atom, group_level=H5Organizer.LEVEL_PATCH)
        with self.hdf5_organizer.pytables[phase]:
            self.database.refresh_atoms()
            self._create_h5array_by_types(phase, filters)
            self._fetch_all_files(phase)
            self.weight_writer.write_classweight_to_db(self.hdf5_organizer, phase)
            # self.database.splits
            self.hdf5_organizer.h5arrays[phase]['split'].append(self.database.splits[phase])
            self.hdf5_organizer.h5arrays[phase]['types'].append(self.types)

    def _create_h5array_by_types(self, phase, filters):
        for category_type in self.types:
            self.hdf5_organizer.build_patch_array(phase, category_type, self.database.atoms[category_type],
                                                  filters)

    def __fetch_data(self, file: str):
        patches_group, valid_tag, type_order, extra_info = self.data_extractor.extract(file)
        patches: Dict[str, Tuple] = {type_name: data for (type_name, data) in zip(type_order, patches_group)}
        return patches, valid_tag, extra_info

    def _fetch_all_files(self, phase):
        for file_id in tqdm(self.database.splits[phase]):
            file = self.database.file_list[file_id]
            patches, valid_tag, extra_info = self.__fetch_data(file)
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
        return self

    def __exit__(self, type_name, value, traceback):
        self.flush()

    def flush(self):
        for phase in self.hdf5_organizer.phases:
            self.hdf5_organizer.flush(phase)


class DataExtractor(DbHelper):
    def __init__(self, database: AbstractDB):
        super().__init__(database)
        self._extract_callable = database.extract_callable
        assert not inspect.isclass(self._extract_callable), "Expect an instantiated callable. " \
                                                            "A class variable is encountered."
        self.meta = type(self)._default_meta(self.database.meta)
        self._meta_load_database(self.database)

    def _meta_load_database(self, database: AbstractDB):
        assert hasattr(database, 'data_shape'), "data_shape uninitialized"
        self.meta.update({'data_shape': database.data_shape})

    @property
    def extract_callable(self):
        return self._extract_callable

    # part of common default values
    @classmethod
    def _default_meta(cls, meta):
        meta['stride_size'] = meta.get('stride_size', 128)
        meta['tissue_area_thresh'] = meta.get('tissue_area_thresh', 0.95)
        meta['interp'] = meta.get('interp', PIL.Image.NONE)
        meta['resize'] = meta.get('resize', 1)
        # meta = SimpleNamespace(**meta)
        return meta

    def extract(self, file) -> Tuple[Tuple[object, ...], Sequence[bool], Sequence[str], object]:
        return self.extract_callable(self, file)


'''
   Weight.
'''


class WeightCollector(DbHelper):
    def __init__(self, database: AbstractDB, extractor: DataExtractor, **kwargs):
        super().__init__(database)
        self.weight_counter_callable = kwargs.get('weight_counter', self.database.weight_counter_callable)
        self._totals = self._new_weight_storage()
        self._extractor = extractor

    @property
    def extractor(self):
        return self._extractor

    @property
    def totals(self):
        return self._totals

    def is_count_weight(self):
        return self.database.enable_weight and self.database.classes is not None

    def weight_accumulate(self, file: str, type_names: Sequence[str],
                          patch_group: Dict[str, object], extra_information):
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

    def write_classweight_to_db(self, hdf5_organizer, phase):
        if self.is_count_weight():
            npixels = hdf5_organizer.build_statistics(phase, 'class_sizes', self.totals)
            npixels[:] = self.totals


class H5Organizer(DbHelper):
    LEVEL_PATCH = 0
    LEVEL_GROUP = 1

    def __init__(self, database: AbstractDB, group_level):
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
    def _get_h5_shapes(chunk_width):
        chunk_shape = [chunk_width]
        h5_shape = [0]
        return h5_shape, chunk_shape

    '''Precondition:h5arrays and pytables initialized/created
    '''

    def build_patch_array(self, phase, category_type, atom, filters=None, expected_rows=None, group_level=None):
        h5_shape, chunk_shape = type(self)._get_h5_shapes(self.database.chunk_width)

        params = {
            'atom': atom,
            'filters': filters,
            'chunkshape': chunk_shape,
        }
        if expected_rows is not None and isinstance(expected_rows, int):
            params['expectedrows'] = expected_rows
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
    def build_statistics(self, phase, category_type, data):
        self.h5arrays[phase][category_type] = self.pytables[phase].create_carray(self.pytables[phase].root,
                                                                                 category_type,
                                                                                 tables.Atom.from_dtype(data.dtype),
                                                                                 data.shape)
        return self.h5arrays[phase][category_type]

    # todo
    def write_data_to_array(self, phase, patches_dict, datasize=None):
        for type_name in self.types:
            self.h5arrays[phase][type_name].append(patches_dict[type_name])
            # write size of data into h5 - todo
            if datasize is not None:
                datasize[type_name] = datasize.get(type_name, 0) + np.asarray(patches_dict[type_name]).shape[0]

    # todo
    def write_file_names_to_array(self, phase, file, patches, category_type="filename"):
        # (patches[self.types[0]] is not None) and (patches[self.types[1]] is not None)
        if patches and all([patches[type_name] is not None for type_name in self.types]):
            # [file for x in range(patches[self.types[0]].shape[0])]
            filename_list = [file] * patches[self.types[0]].shape[0]
            if filename_list:
                self.h5arrays[phase][category_type].append(filename_list)
            else:
                ...  # do nothing. leave it here for tests - todo

    @staticmethod
    def _empty_h5arrays_dict(phases, types) -> Dict[str, Dict]:
        h5arrays = dict()
        for phase in phases:
            h5arrays[phase] = {}
            for type_name in types:
                h5arrays[phase][type_name] = None
        return h5arrays

    def _new_pytables_from_database(self, phases):
        pytables = {}
        for phase in phases:
            pytable_full_path, pytable_dir = self.database.generate_table_name(phase)
            type(self.database).prepare_export_directory(pytable_dir)
            pytables[phase] = tables.open_file(pytable_full_path, mode='w')
        return pytables
