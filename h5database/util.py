from types import SimpleNamespace
from tqdm import tqdm
import numpy as np
import PIL
import tables

class DatabaseNested(object):
	def __init__(self,database):
		self._database = database
		
	@property
	def database(self):
		return self._database

class TaskDispatcher(DatabaseNested):
	def __init__(self,database):
		super().__init__(database)
		self.hdf5_organizer = H5Organizer(self.database,self.database.group_level)
		self.data_extractor = DataExtractor(self.database)
		self.weight_writer =  WeightCollector(self.database, weight_counter = self.database.weight_counter_func)
		self.datasize = {}
	@property
	def types(self):
		return self.database.types
	
	def write(self,phase,filters):
		self.hdf5_organizer.build_patch_array(phase,'filename',self.database.filenameAtom)
		self.hdf5_organizer.build_patch_array(phase,'valid',self.database.validAtom)
		self.hdf5_organizer.build_patch_array(phase,'split',self.database.splitAtom)
		with self.hdf5_organizer.pytables[phase]:	
			self.database.refresh_atoms()
			self._create_h5array_by_types(phase,filters)
			self._fetch_all_files(phase)
			self.weight_writer.write_classweight_to_db(self.hdf5_organizer,phase)		
			#self.database.splits
			self.hdf5_organizer.h5arrays[phase]['split'].append(self.database.splits[phase])
		
	def _create_h5array_by_types(self,phase,filters):
		for category_typetype in self.types:
			self.hdf5_organizer.build_patch_array(phase,category_typetype,self.database.atoms[category_typetype],filters)

	def _fetch_all_files(self,phase):
			patches = {}
			for file_id in tqdm(self.database.splits[phase]):
				file = self.database.filelist[file_id]
				(patches[self.types[0]],patches[self.types[1]],isValid,extra_information) = self.data_extractor.extract(file)
				if (any(list(isValid)) or self.database.write_invalid):
					self.hdf5_organizer.write_data_to_array(phase,patches,self.datasize)
					self.weight_writer.weight_accumulate(file,patches[self.types[0]],patches[self.types[1]],extra_information)
					self.hdf5_organizer.write_file_names_to_array(phase,file,patches,category_type = "filename")
					self.hdf5_organizer.h5arrays[phase]['valid'].append(isValid)

					...
				else:
					#do nothing. leave it blank here for: (1) test. (2) future works
					...	
	def __enter__(self):
		return self
		
	def __exit__(self,type, value, traceback):
		self.flush()
	
	def flush(self):
		for phase in self.hdf5_organizer.phases:
			self.hdf5_organizer.flush(phase)
		
class DataExtractor(DatabaseNested):
	def __init__(self,database):
		super().__init__(database)
		self.meta = self._default_meta(self.database.meta)
	#part of common default values
	def _default_meta(self,meta):
		meta['stride_size'] = meta.get('stride_size',128)
		meta['tissue_area_thresh'] = meta.get('tissue_area_thresh',0.95)
		meta['interp'] = meta.get('interp',PIL.Image.NONE)
		meta['resize'] = meta.get('resize',1)
		return SimpleNamespace(**meta)
	
	def extract(self,file):
		return self.database.patch_pair_extractor_func(self,file)
	


		
'''
	Weight.
'''	
class WeightCollector(DatabaseNested):
	def __init__(self,database,**kwargs):
		super().__init__(database)
		self.weight_counter_func = kwargs.get('weight_counter',self.database.weight_counter_func)
		self._totals = self._new_weight_storage()
	


	@property
	def totals(self):
		return self._totals

	def is_count_weight(self):
		return self.database.enable_weight and self.database.classes is not None
		
	def weight_accumulate(self,file,img,label,extra_information):
		if self.is_count_weight():
			self.count_weight(self._totals,
							  file,
							  img,
							  label,
							  extra_information)	
	
	def _new_weight_storage(self):
		if self.is_count_weight():
			totals = np.zeros(len(self.database.classes))
		else:
			totals = None
		return totals		
		
	def write_classweight_to_db(self,hdf5_organizer,phase):
		if self.is_count_weight():
			npixels = hdf5_organizer.build_statistics(phase,'class_sizes',self.totals)
			npixels[:]=self.totals	

	def count_weight(self,totals,file,img,label,extra_information):
		#weight_counter_func is callback - manually pass the associated database as its 1st arg
		return self.weight_counter_func(self,totals,file,img,label,extra_information)



class H5Organizer(DatabaseNested):
	LEVEL_PATCH = 0
	LEVEL_GROUP = 1
	
	def __init__(self,database,group_level):
		super().__init__(database)
		self._h5arrays = self._empty_h5arrays_dict(self.phases,self.types)
		self._pytables = self._new_pytables_from_database(self.phases)
		self.group_level = group_level
		
	def flush(self,phase):
		for type,h5array in self._h5arrays[phase].items():
			self._h5arrays[phase][type] = []
		self._h5arrays[phase] = []
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
		self.h5arrays[phase][category_type] = create_func(self.pytables[phase].root, category_type,**params)
		return self.h5arrays[phase][category_type]
	#carrays. metadata
	def build_statistics(self,phase,category_type,data):
		self.h5arrays[phase][category_type] = self.pytables[phase].create_carray(self.pytables[phase].root, 
																			 category_type, 
																			 tables.Atom.from_dtype(data.dtype), 
																			 data.shape)
		return self.h5arrays[phase][category_type]																	 
	def write_data_to_array(self,phase,patches_dict,datasize = None):
		for type in self.types:
			self.h5arrays[phase][type].append(patches_dict[type])
			if datasize is not None:
				datasize[type] = datasize.get(type,0)+np.asarray(patches_dict[type]).shape[0]		

	def write_file_names_to_array(self,phase,file,patches,category_type ="filename"):
		if patches and  (patches[self.types[0]] is not None) and (patches[self.types[1]] is not None):
			filename_list = [file for x in range(patches[self.types[0]].shape[0])]
			if filename_list:
				self.h5arrays[phase][category_type].append(filename_list)
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