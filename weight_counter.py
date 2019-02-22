

''' 
	Precondition: totals is defined.   obj.classes is given.

	takes
			totals,file,img, patch/label,  extra_infomration
	returns
		totals 
	
	
	Accumulate totals
'''


'''

'''
def weight_counter_filename(obj,totals,file,img,label,extra_infomration):
	classid=[idx for idx in range(len(obj.classes)) if str(obj.classes[idx]) in file][0]
	totals[classid]+=1
	return totals

def weight_counter_maskpixel(obj,totals,file,img,label,extra_infomration):
	for i,key in enumerate(obj.classes):
		totals[1,i]+=sum(sum(label[:,:,0]==key))
	return totals