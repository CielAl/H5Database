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
				dtype:  data type to be stored in the pytable. Default: UInt8Atom
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
		self.stride_size = kwargs['stride_size']
		
		
		self.numsplit = kwargs.get('numfold',10)
		self.shuffle = kwargs.get('shuffle',True)
		self.tissue_area_thresh = kwargs.get('tissue_ratio',0.95)
		self.patch_pair_extractor_func = kwargs.get('extractor')
		
		self.weight_counter_func = kwargs.get('weight_counter')
		
		self.pattern = kwargs.get('pattern','*.jpg')
		self.interp = kwargs.get('interp',PIL.Image.NONE)
		self.resize = kwargs.get('resize',0.5)
		self.dtype =  kwargs.get('dtype',tables.UInt8Atom())
		self.test_ratio = kwargs.get('test_ratio',0.1)
		
		self.enable_weight = kwargs.get('classweight',False)
		self.classes = kwargs.get('classnames',None)
		
		self.filenameAtom = tables.StringAtom(itemsize=255)

		self.filelist = kwargs.get('filelist',self.get_filelist())
		#for now just take 1 set of train-val shuffle. Leave the n_splits here for future use.
		self.phases = kwargs.get('split',self.init_split())
		self.meta = kwargs.get('meta',{})



	@LazyProperty
	def types(self):
		return ['img','label']
		
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
		phases = {}
		phases[_TRAIN_NAME],phases[_VAL_NAME] = next(self.kfold_split())
		return phases


	
	'''
		Read the file and return (img,label,success,meta).
		Invoke the patch_pair_extractor_func, which is a function handle. So "self" must be explicitly passed to 
		the inputs.
	'''
	def img_label_patches(self,file):
		#patch_pair_extractor_func is a callback: manually pass "self" as its 1st arg
		return self.patch_pair_extractor_func(self,file)
	
	
	
	def count_weight(self,totals,file,img,label,extra_infomration):
		#weight_counter_func is callback - manually pass self as its 1st arg
		return self.weight_counter_func(self,totals,file,img,label,extra_infomration)
	'''
		Generate the Table name from database name.
	'''
	def generate_tablename(self,phase):
		pytable_dir = os.path.join(self.export_dir)
		pytable_fullpath = os.path.join(pytable_dir,"%s_%s%s" %(self.database_name,phase,'.pytable'))
		return pytable_fullpath,pytable_dir
	
	
	def _get_h5_shapes(self,type):
		if np.count_nonzero(self.data_shape[type])!=0:
			chunk_shape= np.append([1],self.data_shape[type])
			h5_shape = np.append([0],self.data_shape[type])
		else:
			chunk_shape  = [1]
			h5_shape = [0]
		return (h5_shape,chunk_shape)
		
		
	
	def _create_h5array_by_types(self,h5arrays,pytable_dict,phase,filters):
		for type in self.types:
			h5_shape,chunk_shape = self._get_h5_shapes(type)
			h5arrays[type]= pytable_dict[phase].create_earray(pytable_dict[phase].root, type, self.dtype,
												  shape= h5_shape, #np.append([0],self.data_shape[type]),
												  chunkshape= chunk_shape,#np.append([1],self.data_shape[type]),
												  filters=filters)
	
	@staticmethod
	def prepare_export_directory(pytable_dir):
			if not os.path.exists(pytable_dir):
				os.makedirs(pytable_dir)	
				
				
	def _write_data_to_db(self,patches_dict,h5arrays,datasize):
		for type in self.types:
			h5arrays[type].append(patches_dict[type])
			datasize[type] = datasize.get(type,0)+np.asarray(patches_dict[type]).shape[0]
	
	def _write_classweight_to_db(self,pytable_dict,phase):
		if self.is_count_weight():
			npixels=pytable_dict[phase].create_carray(pytable_dict[phase].root, 'class_sizes', tables.Atom.from_dtype(self._totals.dtype), self._totals.shape)
			npixels[:]=self._totals	


	def _write_file_names_to_db(self,file,patches,h5arrays):
		if patches and  (patches[self.types[0]] is not None) and (patches[self.types[1]] is not None):
			filename_list = [file for x in range(patches[self.types[0]].shape[0])]
			if filename_list:
				h5arrays["filename"].append(filename_list)
			else:
				...#do nothing. leave it here for tests	
			
			
	def _weight_accumulate(self,file,img,label,extra_infomration):
		if self.is_count_weight():
			self.count_weight(self._totals,
							  file,
							  img,
							  label,
							  extra_inforamtion)	
	
	def _new_weight_storage(self):
		if self.is_count_weight():
			totals = np.zeros(len(self.classes))
		else:
			totals = None
		return totals
	# Tutorial from  https://github.com/jvanvugt/pytorch-unet
	def write_data(self):
		h5arrays = {}
		datasize = {}
		filters=tables.Filters(complevel= 5)
	
		#for each phase create a pytable
		self.tablename = {}
		pytable = {}
		
		self._totals = self._new_weight_storage()
		for phase in self.phases.keys():			
			patches = {}
			pytable_fullpath,pytable_dir = self.generate_tablename(phase)
			
			type(self).prepare_export_directory(pytable_dir)
			
			pytable[phase] = tables.open_file(pytable_fullpath, mode='w')
			h5arrays['filename'] = pytable[phase].create_earray(pytable[phase].root, 'filename', self.filenameAtom, (0,))
			with pytable[phase]:
				self._create_h5array_by_types(h5arrays,pytable,phase,filters)

				for file_id in tqdm(self.phases[phase]):

					file = self.filelist[file_id]					
					(patches[self.types[0]],patches[self.types[1]],isValid,extra_inforamtion) = self.img_label_patches(file)

					if (isValid):
						self._write_data_to_db(patches,h5arrays,datasize)
						self._weight_accumulate(file,patches[self.types[0]],patches[self.types[1]],extra_inforamtion)
						self._write_file_names_to_db(file,patches,h5arrays)
					else:
						#do nothing. leave it blank here for: (1) test. (2) future works
						...
				self._write_classweight_to_db(pytable,phase)		
		return datasize
	
	
	def is_count_weight(self):
		return self.enable_weight and self.classes is not None
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
	
	def peek(self,phase):
		with tables.open_file(self.generate_tablename(phase)[0],'r') as pytable:
			return (getattr(pytable.root,self.types[0]).shape,
			getattr(pytable.root,self.types[1]).shape)
	

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
			self.data_set.phases[_TRAIN_NAME],self.data_set.phases[_VAL_NAME] = self.split[fold]
			self.data_set.write_data()
	