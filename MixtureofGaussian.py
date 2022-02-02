from cgitb import grey
from distutils.archive_util import make_archive
import cv2
import glob
import numpy as np
import math

from PIL import Image
from skimage import img_as_float
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def average(lst):
    return sum(lst) / len(lst)


imdir = 'C:\\Users\\madha\\OneDrive\\Desktop\\Intersection\\'
ext1 = ['png', 'jpg', 'gif'] # Add image formats here

files = []
[files.extend(glob.glob(imdir + '*.' + e)) for e in ext1]
images = [cv2.imread(file) for file in files]

ext2 = ['bmp'] # Ground Truth format
files2 = []
[files2.extend(glob.glob(imdir + '*.' + e)) for e in ext2]
gtimages = [cv2.imread(file) for file in files2]

subtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=50, detectShadows=True)


gtindex=0
gtval=500
mselst=[]
psnrlst=[]
ssimlst=[]


for i in range(len(images)):

    frame=images[i]
    mask=subtractor.apply(frame)
    
    cv2.imshow("Frame",frame)
    cv2.imshow("mask",mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

    if(i==gtval):

        gt=gtimages[gtindex]

        img1=cv2.cvtColor(gt,cv2.COLOR_BGR2RGB)
        img2=cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
        gt_pil=Image.fromarray(img1)
        mask_pil=Image.fromarray(img2)

        mselst.append(mse(img_as_float(mask_pil),img_as_float(gt_pil)))
        psnrlst.append(psnr(img_as_float(mask_pil),img_as_float(gt_pil)))

        ssimlst.append(ssim(img_as_float(mask_pil),img_as_float(gt_pil),multichannel=True,gaussian_weights=True))
        gtindex+=1
        gtval+=15

avgmse=average(mselst)
avgpsnr=average(psnrlst)
avgssim=average(ssimlst)

print("Average MSE: ",avgmse)
print("Average PSNR: ",avgpsnr)
print("Average SSIM: ",avgssim)

cv2.destroyAllWindows()