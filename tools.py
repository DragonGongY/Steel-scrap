import os
import shutil

imgspath = u'/home/dp/Desktop/何豪废钢识别16-23日/2018.10.18hh/'
for imgfolder in os.listdir(imgspath):
    imgpath = os.path.join(imgspath, imgfolder)
    for img in os.listdir(imgpath):
        if '.jpg' in img:
            shutil.copy(imgpath + '/' + img, "/home/dp/2")