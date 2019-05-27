from h5database.util import TaskDispatcher
import glob
import os
# patch extraction and dataset split: Use 1 pair of train-validation to due to limitation of time
from sklearn import model_selection

import tables
from lazy_property import LazyProperty

_TRAIN_NAME = 'train'
_VAL_NAME = 'val'

# key name: img, label
# THIS SCRIPT HANDLES DATA PROCESSING


class Database(object):

    def __init__(self, **kwargs):
        '''Main Class of the Module.

        For each phase (train or val), this class generates a pytable, and saves the extracted patches/labels/masks/...
        into the pytable.
        __getitem__, __len__ are defined to peek the contents. Note, it cannot be directly converted to pytorch database as the
        collections contains both train and val sets.
        Use a tuple index, e.g. database['train',25] to fetch data in certain dataset.
        The scaffolds are inspired from the blog and its corresponding code here:
                http://www.andrewjanowczyk.com/pytorch-unet-for-digital-pathology-segmentation/
        Args:
            **kwargs: Keyword Arguments. Some will be mandatory if "readonly" is false. If mandatory, all except export_dir and database_name are optional.
                Mandatory:
                export_dir (:obj:`str`): location to write the table. Always Mandatory with readonly or not.
                database_name (:obj:`str`): Name of the database. It is also defined as part of the filename of pytables. Always Mandatory with readonly or not.
                filedir (:obj:`str`): basedir of the image data to read.
                data_shape (:obj:`dict`): Keys of the dict are corresponding to the category types, e.g. images/labels/...
                    So far, the keys must be identical to what is defined under the property: Database.types.
                    Values of the dict are the corresponding shape. The type of the shape should be tuple.
                    Example:
                        {
                            'img': (256,256,3),
                            'label':[0], #in case of scalars
                        }
                group_level (int): The level to group the patches together.
                    If 0 is given, there is no grouping. All patches from every images will be stored under the same level (EArray class).
                    If 1 is give, patches from the same image will be stored in a list. Those Lists extracted from different images will be stored in the
                        same level of a VLArray.
                patch_pair_extractor_func (:obj:`function`): The function to generate patches for each image file.
                    Takes:
                        obj: The DataExtractor object which is initialized in _write_data. The DataExtractor has a 'database' attribute which links to the database created
                        and is able to access all information stored in the database. The DataExtractor object also compiles the argument dict "meta" of the database, which stores
                        arbitrary information required in the patch_pair_extractor_func.
                        file: The file path of the data file to extract
                    Returns:
                        img: patch data extracted from the file, or anything fit the defined data_shape.
                        mask/label: Mask, label, or any other information that pairs with img.
                        isValid: A boolean indicating whether to retain the result of the extraction. The results will be saved into HDF5 array iff isValid is True.
                        extra_information: Any other information that should be stored for Class Weight counting.

                Optional:
                    readonly (bool, optional): determine whether it is simply to read pytables (using __getitem__ etc.) or create new ones.
                        If true. All but export_dir and database_name will be optional.
                    chunk_width (int, optional): The width of the chunk in terms of the number of row elements. Default value is 1.
                    numfold (int, optional): Determines the size of fold in the splits. The database class generates only 1 fold of k-fold cross-validation.
                        i.e. it puts  1/numfold of data into validation set and the rest to the training set. See class Kfold for the generation of the complete
                        K-fold cross-validation. Default is 10.
                    shuffle (bool, optional): Determines whether the data will be shuffled in the Cross Validator (k-fold)
                    pattern (:obj:`str`, optional): wildcard pattern of the images. Default is *.png
                    weight_counter_func (:obj:`function`, optional): The function handle to count class weight in-place.
                        Takes:
                            obj: The WeightCollector object which invokes the function (initialized in _write_data)
                                WeightCollector has an attribute "database" that links to the parent database object.
                            totals: The class weight matrix. The accumulation will be done in-place.
                            file: The filepath of the data.
                            img: The patch generated by the extractor.
                            label: The mask/label generated by the extractor.
                            extra_information: The extra information generated by the extractor.
                    classweight (bool, optional): Determines whether to count class weights. If True, weight_counter_func must be given.
                    classnames (:obj:`list`, optional): List of all classnames. The values in the list must implement the to-string conversion, i.e. str(value) must be defined.
                    filelist (:obj:`list`, optional): List of files. If given, it will override the lists generated by the `pattern`.
                    splits (:obj:`dict`, optional): Keys are the phases, i.e. train and val. Values are the array of index (integer) of files in the filelist that are included in the phase.
                        If given, it will override the file split generated by the Cross Validator.
                    meta (:obj:`dict`, optional): The dict that stores all required information for any callbacks, e.g. patch_pair_extractor_func.
                        Default values are given to certain frequently requested information (See class DataExtractor):
                                meta['stride_size'] = meta.get('stride_size',128)
                                meta['tissue_area_thresh'] = meta.get('tissue_area_thresh',0.95)
                                meta['interp'] = meta.get('interp',PIL.Image.NONE)
                                meta['resize'] = meta.get('resize',0.5)
                    row_atom_func (:obj:`tables.atom.MetaAtom`, option): The callable that defines the Atom: the Data type and the shape of the atomic row elements. Default: UInt8Atom
        Raises:
            KeyError if the mandatory inputs is missing
                        interp: the interpolation method, default is PIL.IMAGE.NONE
                resize: the factor of resize the processing, which is 1/downsample_factor.
        '''
        self.KEY_TRAIN = _TRAIN_NAME
        self.KEY_VAL = _VAL_NAME

        self.export_dir = kwargs['export_dir']
        self.database_name = kwargs['database_name']
        self.readonly = kwargs.get('readonly',False)
        if not self.readonly:

            self.filedir = kwargs['filedir']
            self.data_shape = kwargs['data_shape']
            self.validate_shape_key(self.data_shape)
            self.group_level = kwargs['group_level']
            self.patch_pair_extractor_func = kwargs['extractor']

            self.write_invalid = kwargs.get('write_invalid', False)
            self.chunk_width = kwargs.get('chunk_width', 1)
            self.numsplit = kwargs.get('numfold', 10)
            self.shuffle = kwargs.get('shuffle', True)
            self.pattern = kwargs.get('pattern', '*.png')

            self.weight_counter_func = kwargs.get('weight_counter', None)
            self.enable_weight = kwargs.get('classweight', False)
            self.classes = kwargs.get('classnames', None)

            self.filelist = kwargs.get('filelist', self.get_filelist())
            self.splits = kwargs.get('split', self.init_split())

            self.meta = kwargs.get('meta', {})
            self.filenameAtom = tables.StringAtom(itemsize=255)
            self.validAtom = tables.BoolAtom(shape=(), dflt=False)
            self.splitAtom = tables.IntAtom(shape=(), dflt=False)
            self.row_atom_func = kwargs.get('row_atom_func', tables.UInt8Atom)
            self.refresh_atoms()

    def validate_shape_key(self, data_shape):
        assert(sorted(list(data_shape.keys())) == sorted(self.types))

    @staticmethod
    def _init_atoms(row_atom_func, data_shape_dict):
        atoms = {}
        for k, v in data_shape_dict.items():
            atoms[k] = row_atom_func(shape=tuple(v))
        return atoms

    def refresh_atoms(self):
        self._atoms = type(self)._init_atoms(self.row_atom_func,self.data_shape)
        return self._atoms

    @LazyProperty
    def types(self):
        return ['img', 'label']

    @property
    def atoms(self):
        return self._atoms

    @property
    def phases(self):
        return self.splits.keys()
    '''
        Get the list of files by the pattern and location, if not specified by user.
    '''
    def get_filelist(self):
        file_pattern = os.path.join(self.filedir, self.pattern)
        files = glob.glob(file_pattern)
        return files

    def kfold_split(self, stratified_labels=None):

        if stratified_labels is None:
            return iter(model_selection.KFold(n_splits=self.numsplit, shuffle=self.shuffle).split(self.filelist))
        return iter(model_selection.StratifiedKFold(n_splits=self.numsplit, shuffle=self.shuffle)
                    .split(self.filelist, stratified_labels))

    '''
        Initialize the data split and shuffle.
    '''
    def init_split(self, stratified_labels=None):
        splits = dict()
        splits[_TRAIN_NAME], splits[_VAL_NAME] = next(self.kfold_split(stratified_labels=stratified_labels))
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
    def generate_tablename(self, phase):
        pytable_dir = os.path.join(self.export_dir)
        pytable_fullpath = os.path.join(pytable_dir, "%s_%s%s" % (self.database_name, phase, '.pytable'))
        return pytable_fullpath, pytable_dir

    '''
        Return false if no hdf5 found on the export path.
    '''
    def is_instantiated(self, phase):
        file_path = self.generate_tablename(phase)[0]
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
        return self.task_dispatcher.datasize

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
        with tables.open_file(self.generate_tablename(phase)[0], 'r') as pytable:
            image = getattr(pytable.root, self.types[0])[index, ]
            label = getattr(pytable.root, self.types[1])[index, ]
        return image, label

    def size(self, phase):
        with tables.open_file(self.generate_tablename(phase)[0], 'r') as pytable:
            return getattr(pytable.root, self.types[0]).shape[0]

    def __len__(self):
        return sum([self.size(phase) for phase in self.phases])

    def peek(self, phase):
        with tables.open_file(self.generate_tablename(phase)[0], 'r') as pytable:
            return (getattr(pytable.root, self.types[0]).shape,
                    getattr(pytable.root, self.types[1]).shape)

    @staticmethod
    def prepare_export_directory(pytable_dir):
        if not os.path.exists(pytable_dir):
            os.makedirs(pytable_dir)


class Kfold(object):	

    def __init__(self, **kwargs):
        self.numfold = kwargs.get('numfold', 10)
        self.shuffle = kwargs.get('shuffle', True)
        kwargs['numfold'] = self.numfold
        kwargs['shuffle'] = self.shuffle
        self.rootdir = kwargs['export_dir']
        self.data_set = Database(**kwargs)  # init dataset object
        # init split
        self.split = None
        stratified_labels = kwargs.get('stratified_labels', None)
        self.generate_split(stratified_labels=stratified_labels)

    def generate_split(self, stratified_labels=None):
        self.split = list(self.data_set.kfold_split(stratified_labels=stratified_labels))

    def run(self):
        for fold in range(self.numfold):
            # redefine split
            self.data_set.export_dir = os.path.join(self.rootdir, str(fold))
            self.data_set.splits[_TRAIN_NAME], self.data_set.splits[_VAL_NAME] = self.split[fold]
            self.data_set.initialize(force_overwrite=False)
