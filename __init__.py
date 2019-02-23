# THIS SCRIPT HANDLES DATA PROCESSING
''' Requirements: H5 and Table
	keyname: img, label
	
'''
import tables
import glob

import os
import sys

#image io and processing
import cv2
import PIL
import numpy as np

#patch extraction and dataset split: Use 1 pair of train-validation to due to limitation of time
from sklearn import model_selection
import sklearn.feature_extraction.image
from sklearn.feature_extraction.image import extract_patches
import random

from weight_counter import weight_counter_filename

from tqdm import tqdm
from types import SimpleNamespace

from lazy_property import LazyProperty

_TRAIN_NAME = 'train'
_VAL_NAME  = 'val'
class database(object):


	''' The constructor of the class. Some of the logic are inspired from the blog and its corresponding code here:
			http://www.andrewjanowczyk.com/pytorch-unet-for-digital-pathology-segmentation/
		Args:
			Mandatory:
				filedir: base dir of the image
				
				database_name: file name of the database
				export_dir: location to write the table
				data_shape: Dict containing tuples. data_shape[type] is the shape of the patch (type as image or label) 
				stride_size: overlapping of patches.
			Optional:
				pattern: wildcard pattern of the images. Default is *.jpg
				interp: the interpolation method, default is PIL.IMAGE.NONE
				resize: the factor of resize the processing, which is 1/downsample_factor.
				row_atom_func: The callable that defines the Atom: the Data type and the shape of the atomic row elements. Default: UInt8Atom
				test_ratio: ratio of the dataset as test. Default: 0.1
		Raises:
			KeyError if the mandatory inputs is missing
	'''
	def __init__(self,**kwargs):
	
		self.KEY_TRAIN = _TRAIN_NAME
		self.KEY_VAL = _VAL_NAME
		
		self.filedir = kwargs['filedir']
		self.maskdir = kwargs.get('maskdir',None)
		self.database_name = kwargs['database_name']
		self.export_dir = kwargs['export_dir']
		
		self.data_shape = kwargs['data_shape']
		self.validate_shape_key(self.data_shape)
		
		self.group_level = kwargs['group_level']
		
		self.stride_size = kwargs['stride_size']
		
		self.chunk_width = kwargs.get('chunk_width',1)
		self.numsplit = kwargs.get('numfold',10)
		self.shuffle = kwargs.get('shuffle',True)
		self.tissue_area_thresh = kwargs.get('tissue_ratio',0.95)
		self.patch_pair_extractor_func = kwargs.get('extractor')
		
		self.weight_counter_func = kwargs.get('weight_counter')
		
		self.pattern = kwargs.get('pattern','*.jpg')
		self.interp = kwargs.get('interp',PIL.Image.NONE)
		self.resize = kwargs.get('resize',0.5)

		
		
		self.test_ratio = kwargs.get('test_ratio',0.5)
		
		self.enable_weight = kwargs.get('classweight',False)
		self.classes = kwargs.get('classnames',None)
		

		self.filelist = kwargs.get('filelist',self.get_filelist())
		
		self.splits = kwargs.get('split',self.init_split())
		self.meta = kwargs.get('meta',{})

		self.filenameAtom = tables.StringAtom(itemsize=255)
		self.row_atom_func =  kwargs.get('row_atom_func',tables.UInt8Atom)
		self._atoms = self._init_atoms(self.row_atom_func,self.data_shape)
		
		
	def validate_shape_key(self,data_shape):
		assert(sorted(list(self.data_shape.keys())) == sorted(self.types))
	@LazyProperty
	def types(self):
		return ['img','label']
	
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
		file_pattern = os.path.join(self.filedir,self.pattern)
		files=glob.glob(file_pattern)
		return files
	
	def kfold_split(self):
		return iter(model_selection.KFold(n_splits=self.numsplit,shuffle = self.shuffle).split(self.filelist))
	
	'''
		Initialize the data split and shuffle.
	'''
	def init_split(self):
		splits = {}
		splits[_TRAIN_NAME],splits[_VAL_NAME] = next(self.kfold_split())
		return splits


	
	'''
		Read the file and return (img,label,success,meta).
		Invoke the patch_pair_extractor_func, which is a function handle. So "self" must be explicitly passed to 
		the inputs.
	'''
	def img_label_patches(self,file):
		#patch_pair_extractor_func is a callback: manually pass "self" as its 1st arg
		return self.patch_pair_extractor_func(self,file)
	
	
	

	'''
		Generate the Table name from database name.
	'''
	def generate_tablename(self,phase):
		pytable_dir = os.path.join(self.export_dir)
		pytable_fullpath = os.path.join(pytable_dir,"%s_%s%s" %(self.database_name,phase,'.pytable'))
		return pytable_fullpath,pytable_dir
	
	
	def _fetch_all_files(self,phase,hdf5_organizer,datasize):
			patches = {}
			for file_id in tqdm(self.splits[phase]):
				file = self.filelist[file_id]
				(patches[self.types[0]],patches[self.types[1]],isValid,extra_inforamtion) = self.img_label_patches(file)
				if (isValid):
					hdf5_organizer.write_data_to_array(patches,datasize)
					self.weight_writer.weight_accumulate(file,patches[self.types[0]],patches[self.types[1]],extra_inforamtion)
					hdf5_organizer.write_file_names_to_array(file,patches,category_type = "filename")
					
				else:
					#do nothing. leave it blank here for: (1) test. (2) future works
					...	
	'''
		Return false if no hdf5 found on the export path.
	'''	
	def is_instantiated(self,phase):
		file_path = self.generate_tablename(phase)[0]
		return os.path.exists(file_path)
	'''
		non-overridable
	'''	
	def initialize(self):
		if (not self.is_instantiated(_TRAIN_NAME)) or (not self.is_instantiated(_VAL_NAME)):
			self.write_data()
			
	'''
		Slice the chunk of patches out of the database.
		Precondition: Existence of pytable file. Does not require to call "write_data" as long as pytables exist
		Args:
			phase_index_tuple: ('train/val',index)
		Returns:
			images, labels(ground truth)
	'''
	def __getitem__(self, phase_index_tuple):
		phase,index = phase_index_tuple
		with tables.open_file(self.generate_tablename(phase)[0],'r') as pytable:
			image = getattr(pytable.root,self.types[0])[index,]
			label = getattr(pytable.root,self.types[1])[index,]
		return image,label

	def size(self,phase):
		with tables.open_file(self.generate_tablename(phase)[0],'r') as pytable:
			return getattr(pytable.root,self.types[0]).shape[0]

	def __len__(self):
		return sum([self.size(phase) for phase in self.phases])
		
	def peek(self,phase):
		with tables.open_file(self.generate_tablename(phase)[0],'r') as pytable:
			return (getattr(pytable.root,self.types[0]).shape,
			getattr(pytable.root,self.types[1]).shape)


		
		
	
	def _create_h5array_by_types(self,hdf5_organizer,phase,filters):
		for category_typetype in self.types:
			hdf5_organizer.build_patch_array(phase,category_typetype,self.atoms[category_typetype],filters)
	
	
	@staticmethod
	def prepare_export_directory(pytable_dir):
			if not os.path.exists(pytable_dir):
				os.makedirs(pytable_dir)	
				
				

	



			
			

	# Tutorial from  https://github.com/jvanvugt/pytorch-unet
	
	def write_data(self):
		h5arrays = {}
		datasize = {}
		filters=tables.Filters(complevel= 5)
		
		hdf5_organizer = self.H5Organizer(self,self.group_level)
		self.weight_writer = self.WeightCollector(self, weight_counter = self.weight_counter_func)
		
		for phase in self.phases:			
			patches = {}
			hdf5_organizer.build_patch_array(phase,'filename',self.filenameAtom)
			with hdf5_organizer.pytables[phase]:	
				self._create_h5array_by_types(hdf5_organizer,phase,filters)
				self._fetch_all_files(phase,hdf5_organizer,datasize)
				self.weight_writer.write_classweight_to_db(hdf5_organizer,phase)		
		return datasize
	
	
	

			
	'''
		Weight.
	'''	
	class WeightCollector(object):
		def __init__(self,database,**kwargs):
			self._database = database
			self.weight_counter_func = kwargs.get('weight_counter',self.database.weight_counter_func)
			self._totals = self._new_weight_storage()
		
		@property
		def database(self):
			return self._database

		@property
		def totals(self):
			return self._totals

		def is_count_weight(self):
			return self.database.enable_weight and self.database.classes is not None
			
		def weight_accumulate(self,file,img,label,extra_infomration):
			if self.is_count_weight():
				self.count_weight(self._totals,
								  file,
								  img,
								  label,
								  extra_inforamtion)	
		
		def _new_weight_storage(self):
			if self.is_count_weight():
				totals = np.zeros(len(self.database.classes))
			else:
				totals = None
			return totals		
			
		def write_classweight_to_db(self,hdf5_organizer,phase):
			if self.is_count_weight():
				npixels = hdf5_organizer.build_statistics(phase,'class_sizes',data)
				npixels[:]=self._totals	

		def count_weight(self,totals,file,img,label,extra_infomration):
			#weight_counter_func is callback - manually pass the associated database as its 1st arg
			return self.weight_counter_func(self.database,totals,file,img,label,extra_infomration)
	
	def _init_atoms(self,row_atom_func,data_shape_dict):
		atoms = {}
		for k,v in data_shape_dict.items():
				atoms[k] = row_atom_func(shape = tuple(v))
		return atoms
	
	class H5Organizer(object):
		LEVEL_PATCH = 0
		LEVEL_GROUP = 1
		
		def __init__(self,database,group_level):
			self._database = database
			self._h5arrays = self._empty_h5arrays_dict(self.phases,self.types)
			self._pytables = self._new_pytables_from_database(self.phases)
			self.group_level = group_level
			
		@property
		def database(self):
			return self._database		
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
		def filenameAtom(self):
			return self.database.filenameAtom
	
		@property
		def phases(self):
			return self.database.phases
		
		@property
		def types(self):
			return self.database.types
					
		def _get_h5_shapes(self,chunk_width):
			chunk_shape  = [chunk_width] 
			h5_shape = [0]
			return (h5_shape,chunk_shape)

		'''Precondition:h5arrays and pytables initialized/created
		'''
		def build_patch_array(self,phase,category_type,atom,filters=None,expected_rows = None):
			h5_shape,chunk_shape = self._get_h5_shapes(self.database.chunk_width)
			
			params = {
				'atom':atom,
				'filters':filters,
				'chunkshape':chunk_shape,
			}
			if expected_rows is not None and isinstance(expected_rows,int):
				params['expectedrows'] = expected_rows
			
			if self.group_level is type(self).LEVEL_PATCH:
				create_func = self.pytables[phase].create_earray
				params['shape'] = h5_shape
			elif self.group_level is type(self).LEVEL_GROUP:
				create_func = self.pytables[phase].create_vlarray
			else:
				raise ValueError('Invalid Group Level:'+str(self.group_level))
			self.h5arrays[category_type] = create_func(self.pytables[phase].root, category_type,**params)
			return self.h5arrays[category_type]
		#carrays. metadata
		def build_statistics(self,phase,category_type,data):
			self.h5arrays[category_type] = self.pytables[phase].create_carray(pytable_dict[phase].root, 
																				 category_type, 
																				 tables.Atom.from_dtype(data.dtype), 
																				 data.shape)
			return self.h5arrays[category_type]																	 
		def write_data_to_array(self,patches_dict,datasize = None):
			for type in self.types:
				self.h5arrays[type].append(patches_dict[type])
				if datasize is not None:
					datasize[type] = datasize.get(type,0)+np.asarray(patches_dict[type]).shape[0]		

		def write_file_names_to_array(self,file,patches,category_type ="filename"):
			if patches and  (patches[self.types[0]] is not None) and (patches[self.types[1]] is not None):
				filename_list = [file for x in range(patches[self.types[0]].shape[0])]
				if filename_list:
					self.h5arrays[category_type].append(filename_list)
				else:
					...#do nothing. leave it here for tests	

		def _empty_h5arrays_dict(self,phases,types):
			h5arrays = {}
			for phase in phases:
				h5arrays[phase] = {}
				for type in types:
					h5arrays[phase][type] = None
			return h5arrays
		
		def _new_pytables_from_database(self,phases):
			pytables = {}
			for phase in phases:
				pytable_fullpath,pytable_dir = self.database.generate_tablename(phase)
				type(self.database).prepare_export_directory(pytable_dir)
				pytables[phase] = tables.open_file(pytable_fullpath, mode='w')
			return pytables

class kfold(object):	


	def __init__(self,**kwargs):
		self.numfold = kwargs.get('numfold',10)
		self.shuffle = kwargs.get('shuffle',True)
		kwargs['numfold'] = self.numfold
		kwargs['shuffle'] = self.shuffle
		self.rootdir = kwargs['export_dir']
		self.data_set = database(**kwargs) #init dataset object
		self.split = list(self.data_set.kfold_split())

	def run(self):
		for fold in range(self.numfold):
			#redifine split
			self.data_set.export_dir = os.path.join(self.rootdir,str(fold))
			self.data_set.splits[_TRAIN_NAME],self.data_set.splits[_VAL_NAME] = self.split[fold]
			self.data_set.write_data()
	