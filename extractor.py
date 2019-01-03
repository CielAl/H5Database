from sklearn import model_selection
from sklearn.feature_extraction.image import extract_patches
import cv2

def srcnn_img_label_pair(obj,file):
	img = cv2.imread(file,cv2.COLOR_BGR2RGB)
	img_down = cv2.resize(img,(0,0),fx=obj.resize,fy=obj.resize, interpolation=obj.interp)
	#the shape is (y,x) while cv2.resize requires (x,y)
	img_down = cv2.resize(img_down,(img.shape[1],img.shape[0]),interpolation=obj.interp)
	return img,img_down

def generate_patch(obj,image,type = 'img'):
	patches_label= extract_patches(image,obj.data_shape[type],obj.stride_size)
	patches_label = patches_label.reshape((-1,)+obj.data_shape[type])
	return patches_label

#image label
def extractor_super_resolution(obj,file):
	img_truth,img_down = srcnn_img_label_pair(obj,file)
	image = generate_patch(obj,img_down,'img')
	label = generate_patch(obj,img_truth,'label')
	return (image,label,True)

	

	
#############






def extractor_melanoma(obj,file):
	classname = ["BCC", "SCC"]
	image_whole = cv2.cvtColor(cv2.imread(fname),cv2.COLOR_BGR2RGB)
	image_whole = cv2.resize(image_whole,(0,0),fx=obj.resize,fy=obj.resize, interpolation=PIL.Image.NONE)
	data_image  = generate_patch(obj,image_whole,type = 'img')
	data_label = [classid for x in range(data_image.shape[0])]
	return (image,label,True)