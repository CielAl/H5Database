import os
import re

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


def weight_counter_filename(obj, totals, file, img, label, extra_infomration):
    basename = os.path.basename(file)
    class_list = [idx for idx in range(len(obj.database.classes)) if
                  re.search(str(obj.database.classes[idx]), basename, re.IGNORECASE)]
    class_id = class_list[0]
    totals[class_id] += len(label)
    return totals


def weight_counter_maskpixel(obj, totals, file, img, label, extra_infomration):
    for i, key in enumerate(obj.database.classes):
        totals[1, i] += sum(sum(label[:, :, 0] == key))
    return totals
