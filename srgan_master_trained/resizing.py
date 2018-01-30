import os, sys
import PIL
from PIL import Image

images = os.listdir('./BioIDFace_HR')

for imageName in images:
    img = Image.open('./BioIDFace_HR/'+imageName)
    imageName = os.path.splitext(imageName)[0]
    imageName = imageName + '.png'
    
    newSize = (int(img.size[0]/4),int(img.size[1]/4))
    normSize = (newSize[0]*4, newSize[1]*4)

    imgNorm = img.resize(normSize, Image.ANTIALIAS)
    imgNorm.save('./BioIDFace_HR/'+imageName)
    
    imgRed = img.resize(newSize, Image.ANTIALIAS)
    imgRed.save('./BioIDFace_LR/'+imageName)

    print(imageName+' resized.')