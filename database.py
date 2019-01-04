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

from tqdm import tqdm
from types import SimpleNamespace


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
		self.filedir = kwargs['filedir']
		self.maskdir = kwargs.get('maskdir',None)
		self.database_name = kwargs['database_name']
		self.export_dir = kwargs['export_dir']
		
		self.data_shape = kwargs['data_shape']
		self.stride_size = kwargs['stride_size']
		
		self.tissue_area_thresh = kwargs.get('tissue_ratio',0.95)
		self.patch_pair_extractor = kwargs.get('extractor')
		self.pattern = kwargs.get('pattern','*.jpg')
		self.interp = kwargs.get('interp',PIL.Image.NONE)
		self.resize = kwargs.get('resize',0.5)
		self.dtype =  kwargs.get('dtype',tables.UInt8Atom())
		self.test_ratio = kwargs.get('test_ratio',0.1)
		
		self.enable_weight = kwargs.get('classweight',False)
		self.class_names = kwargs.get('classnames',None)
		
		self.filenameAtom = tables.StringAtom(itemsize=255)

		self.filelist = kwargs.get('filelist',self.get_filelist())
		#for now just take 1 set of train-val shuffle. Leave the n_splits here for future use.
		self.phases = kwargs.get('split',self.init_split())
		self.meta = kwargs.get('meta',{})
		self.types = ['img','label']

	'''
		Get the list of files by the pattern and location, if not specified by user.
	'''
	def get_filelist(self):
		file_pattern = os.path.join(self.filedir,self.pattern)
		files=glob.glob(file_pattern)
		return files
	'''
		Initialize the data split and shuffle.
	'''
	def init_split(self):
		phases = {}
		phases['train'],phases['val'] = next(iter(model_selection.ShuffleSplit(n_splits=10,test_size=self.test_ratio).split(self.filelist)))
		return phases


	
	'''
		Read the file and return (img,label,success,meta).
		Invoke the patch_pair_extractor, which is a function handle. So "self" must be explicitly passed to 
		the inputs.
	'''
	def img_label_patches(self,file):
		return self.patch_pair_extractor(self,file)
	
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
	# Tutorial from  https://github.com/jvanvugt/pytorch-unet
	def write_data(self):
		h5arrays = {}
		datasize = {}
		filters=tables.Filters(complevel= 5)
	
		#for each phase create a pytable
		self.tablename = {}
		pytable = {}
		patches = {}
		totals = np.zeros(len(self.class_names))
		for phase in self.phases.keys():			
			#self.tablename[phase] = pytable_fullpath
			pytable_fullpath,pytable_dir = self.generate_tablename(phase)
			if not os.path.exists(pytable_dir):
				os.makedirs(pytable_dir)
			pytable[phase] = tables.open_file(pytable_fullpath, mode='w')
			#datasize[phase] = pytable
			h5arrays['filename'] = pytable[phase].create_earray(pytable[phase].root, 'filename', self.filenameAtom, (0,))

			for type in self.types:
				h5_shape,chunk_shape = self._get_chunk_shape(type)
				h5arrays[type]= pytable[phase].create_earray(pytable[phase].root, type, self.dtype,
													  shape= h5_shape, #np.append([0],self.data_shape[type]),
													  chunkshape= chunk_shape,#np.append([1],self.data_shape[type]),
													  filters=filters)
			#cv2.COLOR_BGR2RGB
			for file_id in tqdm(self.phases[phase]):
				#img as label,
				file = self.filelist[file_id]
				
				(patches[self.types[0]],patches[self.types[1]],isValid) = self.img_label_patches(file)
				
				
				if (isValid):
					for type in self.types:
						h5arrays[type].append(patches[type])
						datasize[type] = datasize.get(type,0)+patches[type].shape[0]
				if self.enable_weight and self.class_names is not None:
					classid=[idx for idx in range(len(obj.class_names)) if obj.class_names[idx] in file][0]
					totals[classid]+=1
			
			h5arrays["filename"].append([file for x in range(patches[self.types[0]].shape[0])])
			
			if self.enable_weight:
				npixels=hdf5_file.create_carray(pytable[phase].root, 'classsizes', tables.Atom.from_dtype(totals.dtype), totals.shape)
				npixels[:]=totals
			
			for k,v in pytable.items():
				v.close()
		return datasize
	
	

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
		if (not self.is_instantiated('train')) or (not self.is_instantiated('val')):
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
			image = pytable.root.img[index,]
			label = pytable.root.label[index,]
		return image,label

	def size(self,phase):
		with tables.open_file(self.generate_tablename(phase)[0],'r') as pytable:
			return pytable.root.img.shape[0]
	
	def peek(self,phase):
		with tables.open_file(self.generate_tablename(phase)[0],'r') as pytable:
			return pytable.root.img.shape,pytable.root.label.shape
