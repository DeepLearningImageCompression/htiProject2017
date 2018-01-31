import os, sys
import PIL
from PIL import Image
from skimage import color
from skimage import io
import scipy.misc
import numpy as np


a = 0
for arg in sys.argv:
    print(arg)
    if a == 2:
        imageName = arg
    elif a == 1:
        road = arg
    a += 1


img = Image.open(road+"/"+imageName)
# print(img)

(imax, jmax) = np.shape(img)
print(imax, jmax)
i_max = imax - imax % 8
j_max = jmax - jmax % 8

for i in range(0, i_max, 8):
    for j in range(0, j_max, 8):
        box = (i, j, i+8, j+8)
        area = img.crop(box)
        # area.show()

        # crop = img[i:i+3][j:j+3][0:3]
        # print(crop)
        # scipy.misc.imsave("crop_8/"+i+"_"+j+imageName[0:-3]+"png", crop)

        print(area.size)
        area.save("crop_8/"+str(i)+"_"+str(j)+imageName[0:-3]+"png", "PNG")
    print(i,j)
