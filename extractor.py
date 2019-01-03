from sklearn import model_selection
from sklearn.feature_extraction.image import extract_patches
import cv2

def srcnn_img_label_pair(obj,file):
	img = cv2.imread(file,cv2.COLOR_BGR2RGB)
	img_down = cv2.resize(img,(0,0),fx=obj.resize,fy=obj.resize, interpolation=obj.interp)
	#the shape is (y,x) while cv2.resize requires (x,y)
	img_down = cv2.resize(img_down,(img.shape[1],img.shape[0]),interpolation=obj.interp)
	return img,img_down

def generate_patch(obj,image):
	patches_label= extract_patches(image,obj.patch_shape,obj.stride_size)
	patches_label = patches_label.reshape((-1,)+obj.patch_shape)
	return patches_label


def extractor_super_resolution(obj,file):
	img_truth,img_down = srcnn_img_label_pair(obj,file)
	patches_down = generate_patch(obj,img_down)
	patches_truth = generate_patch(obj,img_truth)
	return (patches_truth,patches_down,True)