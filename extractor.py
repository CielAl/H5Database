from sklearn import model_selection
from sklearn.feature_extraction.image import extract_patches
import cv2
from skimage.color import rgb2gray
import numpy as np 
import os

import PIL
import skimage
import re
'''
	takes	Extractor object, file
	return img, mask(or label), isvalid, extra_inforamtion
'''

def srcnn_img_label_pair(obj,file):
	img = cv2.imread(file,cv2.COLOR_BGR2RGB)
	img_down = cv2.resize(img,(0,0),fx=obj.meta.resize,fy=obj.meta.resize, interpolation=obj.meta.interp)
	#the shape is (y,x) while cv2.resize requires (x,y)
	img_down = cv2.resize(img_down,(img.shape[1],img.shape[0]),interpolation=obj.meta.interp)
	return img,img_down

def generate_patch(obj,image,type = 'img'):
	patches_label= extract_patches(image,obj.database.data_shape[type],obj.meta.stride_size)
	patches_label = patches_label.reshape((-1,)+obj.database.data_shape[type])
	return patches_label

#image label
def extractor_super_resolution(obj,file):
	img_truth,img_down = srcnn_img_label_pair(obj,file)
	image = generate_patch(obj,img_down,'img')
	label = generate_patch(obj,img_truth,'label')
	return (image,label,True,None)

	

	
#############



def getBackground(img_gray,params = {}):
	img_laplace = np.abs(skimage.filters.laplace(img_gray))
	mask = (skimage.filters.gaussian(img_laplace, sigma=params.get("variance", 5)) <= params.get("smooth_thresh", 0.03))
	background = (mask!=0) * img_gray
	background[mask==0] = 1# background[mask_background].mean()
	return background,mask


def qualification_no_background_helper(patch,tissue_threshold_ratio):
	return (not patch.size<=0) and (np.count_nonzero(patch)/patch.size>=(tissue_threshold_ratio) )

def patch_qualification(patch_sanitized,tissue_threshold_ratio = 0.95):
	idx_qualified = []
	for (idx,patch) in enumerate(patch_sanitized):
		#patch with 95%(default) non-zero pixels - add idx to patch candididates
		if qualification_no_background_helper(patch,tissue_threshold_ratio):
			idx_qualified.append(idx)
	return patch_sanitized[idx_qualified]
def background_sanitize(image,params={}):
	img_gray = rgb2gray(image)
	background,mask = getBackground(img_gray,params) #- pixel 1/True is the background part
	image[mask==1] = 0
	return image

def get_mask_name_default(obj,img_name_pattern):
	img_name = img_name_pattern[0]
	suffix = getattr(obj.meta,'mask_suffix','_mask')
	type = getattr(obj.meta,'mask_ext','png')
	full_maskname = f"{img_name}{suffix}.{type}"
	return full_maskname
	
#note: no multi
def mask_screening(obj,file,image):
	basename = os.path.basename(file) 
	filename_pattern = os.path.splitext(basename)
	mask_name_func = getattr(obj.meta,'mask_name_getter',get_mask_name_default)
	maskname = mask_name_func(obj,filename_pattern)
	mask_fullpath = os.path.join(obj.meta.mask_dir,maskname)
	mask = cv2.imread(mask_fullpath).astype(int)
	#print(mask.shape)
	#print(image.shape)
	image[mask<=0] = 0
	return image

def extractor_patch_classification(obj,file):
	basename = os.path.basename(file) 
	classid=[idx for idx in range(len(obj.database.classes)) if re.search(obj.database.classes[idx],basename,re.IGNORECASE)][0]
	image_whole = cv2.cvtColor(cv2.imread(file),cv2.COLOR_BGR2RGB)
	image_whole = cv2.resize(image_whole,(0,0),fx=obj.meta.resize,fy=obj.meta.resize, interpolation=PIL.Image.NONE)
	if obj.meta.mirror_padding and min(image_whole.shape[0:-1])<= obj.database.data_shape['img'][0]:
		mirror_pad_size = obj.meta.mirror_pad_size
		image_whole = np.pad(image_whole, [(mirror_pad_size, mirror_pad_size), (mirror_pad_size, mirror_pad_size), (0, 0)], mode="reflect")
	#discard patches that are too small - mistakes of annotations?
	if image_whole.shape[0]<obj.database.data_shape['img'][0] or image_whole.shape[1]<obj.database.data_shape['img'][1]:
		return (None,None,False,None)

	if hasattr(obj.meta,'mask_dir') and obj.meta.mask_dir is not None:
		#combine masks
		image_whole = mask_screening(obj,file,image_whole)

	#make background pixel strictly 0
	image_whole_sanitized = background_sanitize(image_whole)
	data_image_sanitized  = generate_patch(obj,image_whole_sanitized,type = 'img')
	
	data_image_qualified= patch_qualification(data_image_sanitized,obj.meta.tissue_area_thresh) 
	data_label = [classid for x in range(data_image_qualified.shape[0])]
	return (data_image_qualified,data_label,True,None)

