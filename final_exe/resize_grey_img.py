import os, sys
import PIL
from PIL import Image
from skimage import color
from skimage import io
import scipy.misc
import numpy as np

i = 0
for arg in sys.argv:
    print(arg)
    if i == 2:
        imageName = arg
    elif i == 1:
        road = arg
    i += 1

# imageName = Image.open(imageName)
# imageName.save('img.png')

# imageName = Image.open(imageName).convert('LA')
# imageName.save('greyscale.jpg')

# print(io.imread(imageName))

# img = color.rgb2gray(io.imread(imageName))
# print(img)
# img.save('greyscale.jpg')

img = io.imread(road+"/"+imageName, as_grey=True)
img = img*256
img = np.int_(img)
# print(img)
# img.save('greyscale.jpg')
# img.save_image(imageName,'./greyscale.jpg')
scipy.misc.imsave('greyscale.jpg', img)

img = Image.open('greyscale.jpg')

# img = Image.open('greyscale.jpg')

 # imageName = os.path.splitext('greyscale.jpg')[0]

newSize = (int(img.size[0]/4),int(img.size[1]/4))
normSize = (newSize[0]*4, newSize[1]*4)

# imageIni = Image.open(imageName)
imgNorm = img.resize(normSize, Image.ANTIALIAS)
imgNorm.save("test_hr/"+imageName)

imgRed = img.resize(newSize, Image.ANTIALIAS)
imgRed.save('resize_grey.png')

print(imageName+' resized')
