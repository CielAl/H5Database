

''' 
	Precondition: totals is defined.   obj.database.classes is given.

	takes
			WeightCollector, totals,file,img, patch/label,  extra_infomration
	returns
		totals 
	
	
	Accumulate totals
'''


'''

'''
def weight_counter_filename(obj,totals,file,img,label,extra_infomration):
	class_list = [idx for idx in range(len(obj.database.classes)) if str(obj.database.classes[idx]) in file]
	classid= class_list[0]
	totals[classid]+= len(class_list)
	return totals

def weight_counter_maskpixel(obj,totals,file,img,label,extra_infomration):
	for i,key in enumerate(obj.database.classes):
		totals[1,i]+=sum(sum(label[:,:,0]==key))
	return totals