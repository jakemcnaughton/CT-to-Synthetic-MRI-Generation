import os
import nibabel as nib
import tensorflow as tf
#import keras
import numpy as np
import keras.backend as K
import math
from skimage.metrics import structural_similarity

fakepath = "/hpc/jmcn735/Unet/CropFinal2/test/MR"
truepath = "/hpc/jmcn735/Unet/300FinalUnet5e-5b1"

fakes=[]

maes=[]
mses=[]
psnrs=[]
ssims=[]
names=[]
length=176
height=192
width=176
trues=[]
patients=[]
def PSNR(y_true, y_pred):
    max_pixel = 255
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))
    


for dirName, subdirList, fileList in os.walk(fakepath):
      for filename in fileList:
          if ".nii" in filename:
            filetoadd=np.array(nib.load(os.path.join(fakepath,filename)).get_fdata())
            fakes.append(filetoadd)
            
            
            
for dirName, subdirList, fileList in os.walk(truepath):
      for filename in fileList:
          if ".nii" in filename:
            names.append(filename)
            filetoadd=np.array(nib.load(os.path.join(truepath,filename)).get_fdata())
            trues.append(filetoadd)


for i in range(len(trues)):
    mae=0
    mse=0
    p=0
    fake = fakes[i]
    diff = fakes[i] - trues[i]
    for x in range(length):
      for y in range(height):
        for z in range(width):
          if fake[x,y,z]!=0:
            p+=1
            mse+=diff[x,y,z]**2
            mae+=np.abs(diff[x,y,z])
    
    psnr=10.0 * (1.0 / math.log(10)) * K.log((255** 2) / (mse/p))
    maes.append(mae/p)
    mses.append(mse/p)
    psnrs.append(psnr)
    SSIM = structural_similarity(fakes[i],trues[i], data_range=255, full=True)
    ssims.append(SSIM)
print(np.mean(mses))   
print(np.mean(maes)) 
print(np.mean(psnrs))  
print(ssims[0].numpy())
nib.save(nib.Nifti1Image(ssims[0].numpy(), np.eye(4)), "/hpc/jmcn735/difftensssim.nii.gz")
"""diff = fakes[i] - trues[i]
mae = np.mean(np.abs(diff))
mse = np.mean(np.square(diff))

psnr = PSNR(trues[i], fakes[i])
print(names[i])
print(mae)
print(mse)
print(SSIM)
print(psnr)
maes.append(mae)
mses.append(mse)
ssims.append(SSIM)
psnrs.append(psnr)"""
    
#print("Test Results for {} \n MAE: {}, MSE: {}, SSIM: {}, PSNR: {}".format(Model, np.mean(maes), np.mean(mses), np.mean(ssims), np.mean(psnrs)))