import os
import nibabel as nib
import tensorflow as tf
#import keras
import numpy as np
import keras.backend as K
import math
#import torch
#from torchmetrics import StructuralSimilarityIndexMeasure
from skimage.metrics import structural_similarity as ssim

fakepath = "/hpc/jmcn735/Unet/Final/test/MR"
truepath = "/hpc/jmcn735/Unet/300Unet++96patch5e5b4patchify"

fakes=[]

maes=[]
ssims_total=[]
mses=[]
psnrs=[]
ssims=[]
length=176
height=192
width=176
trues=[]
patients=[]
def PSNR(y_true, y_pred):
    max_pixel = 255
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))
    


for dirName, subdirList, fileList in os.walk(fakepath):
      for filename in sorted(fileList):
          if ".nii" in filename:
            filetoadd=nib.load(os.path.join(fakepath,filename)).get_fdata()
            filetoadd=filetoadd[:,12:204,:]
            #filetoadd=np.pad(filetoadd, ((5, 6),(0,0),(5, 6)))
            filetoadd=255*filetoadd[0:176,:,0:176]
            fakes.append(filetoadd)
            
            
            
for dirName, subdirList, fileList in os.walk(truepath):
      for filename in sorted(fileList):
          if ".nii" in filename:
            filetoadd=nib.load(os.path.join(truepath,filename)).get_fdata()
            #filetoadd=filetoadd[:,12:204,:]
            #filetoadd=np.pad(filetoadd, ((5, 6),(0,0),(5, 6)))
            #filetoadd=filetoadd[8:184,:,8:184]
            #filetoadd=np.swapaxes(filetoadd, 0, 2)
            #filetoadd = np.rot90(filetoadd, 1, (0,1))
            #filetoadd = np.flip(filetoadd, 0)
            filetoadd=filetoadd[5:181,:,5:181]
            #filetoadd=filetoadd[:,12:204,:]
            #filetoadd=np.pad(filetoadd, ((8, 8),(0,0),(8, 8)))
            trues.append(filetoadd)


for i in range(len(trues)):
    mae=0
    mse=0
    p=0
    ssim_total=0
    fake = fakes[i]
    diff = fakes[i] - trues[i]
    fake = fakes[i]
    true = trues[i]
    #print(fake.unsqueeze(0).shape)
    SSIM=ssim(fake, true, data_range=255, win_size=11, full=True, gaussian_weights=True)
    SSIM=SSIM[1]
    
    for x in range(length):
      for y in range(height):
        for z in range(width):
          if fake[x,y,z]!=0:
            p+=1
            mse+=diff[x,y,z]**2
            mae+=np.abs(diff[x,y,z])
            ssim_total+=SSIM[x,y,z]
    psnr=10.0 * (1.0 / math.log(10)) * K.log(255**2 / (mse/p))
    maes.append(mae/p)
    mses.append(mse/p)
    psnrs.append(psnr)
    SSIM=ssim(fake, true, data_range=255, win_size=11, gaussian_weights=True)
    print(SSIM)
    print(tf.image.ssim(fake, true, max_val=255))
    #SSIM=nib.Nifti1Image(SSIM[1], np.eye(4))
    #nib.save(SSIM,os.path.join("/hpc/jmcn735/Unet/SSIMS/",str(i)+".nii.gz"))
    ssims.append(ssim_total/p)
    ssims_total.append(SSIM)

print(np.mean(mses))   
print(np.mean(maes)) 
print(np.mean(psnrs))  
print(np.mean(ssims))
#for ssim in ssims:
#  nib.save(stripped_ct,os.path.join(ctextrdir,patient+".nii.gz"))
#print(ssims)
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
model =  "UnetV2 1"
print("Test Results for {} \n MAE: {} pm {}, MSE: {} pm {}, SSIM: {} pm {}, PSNR: {} pm {}, Total SSIM: {} pm {}".format(model, np.mean(maes), np.std(maes), np.mean(mses), np.std(mses), np.mean(ssims), np.std(ssims),np.mean(psnrs), np.std(psnrs), np.mean(ssims_total), np.std(ssims_total)))