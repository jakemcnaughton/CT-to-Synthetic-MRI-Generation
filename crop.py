import os
import numpy as np
import nibabel as nib
trainpath="/hpc/jmcn735/Unet/Final/train/CT"
targetpath="/hpc/jmcn735/Unet/Final/train/MR"
valpath1="/hpc/jmcn735/Unet/Final/val/CT"
valpath2="/hpc/jmcn735/Unet/Final/val/MR"
testctpath = "/hpc/jmcn735/Unet/Final/test/CT"
testmrpath = "/hpc/jmcn735/Unet/Final/test/MR" 
"""    
  
for dirName, subdirList, fileList in os.walk(trainpath):
    for filename in fileList:
        filetoadd=np.array(nib.load(os.path.join(trainpath,filename)).get_fdata())
        filetoadd=filetoadd[0:176,12:204,0:176] 
        filetoadd=nib.Nifti1Image(filetoadd, np.eye(4))
        nib.save(filetoadd, "/hpc/jmcn735/Unet/CropFinal/train/CT/"+filename)
        
for dirName, subdirList, fileList in os.walk(targetpath):
    for filename in fileList:
        filetoadd=255*np.array(nib.load(os.path.join(targetpath,filename)).get_fdata())
        filetoadd=filetoadd[0:176,12:204,0:176]
        filetoadd=nib.Nifti1Image(filetoadd, np.eye(4))
        nib.save(filetoadd, "/hpc/jmcn735/Unet/CropFinal/train/MR/"+filename)
        
for dirName, subdirList, fileList in os.walk(valpath1):
    for filename in fileList:
        filetoadd=np.array(nib.load(os.path.join(valpath1,filename)).get_fdata())
        filetoadd=filetoadd[0:176,12:204,0:176]
        filetoadd=nib.Nifti1Image(filetoadd, np.eye(4))  
        nib.save(filetoadd, "/hpc/jmcn735/Unet/CropFinal/val/CT/"+filename)          
        
for dirName, subdirList, fileList in os.walk(valpath2):
    for filename in fileList:
        filetoadd=255*np.array(nib.load(os.path.join(valpath2,filename)).get_fdata())
        filetoadd=filetoadd[0:176,12:204,0:176]
        filetoadd=nib.Nifti1Image(filetoadd, np.eye(4))
        nib.save(filetoadd, "/hpc/jmcn735/Unet/CropFinal/val/MR/"+filename)
"""
for dirName, subdirList, fileList in os.walk(testctpath):
    for filename in fileList:
        filetoadd=np.array(nib.load(os.path.join(testctpath,filename)).get_fdata())
        filetoadd=filetoadd[0:176,12:204,0:176]
        filetoadd=nib.Nifti1Image(filetoadd, np.eye(4))
        nib.save(filetoadd, "/hpc/jmcn735/Unet/CropFinal2/test/CT/"+filename)
        
        
for dirName, subdirList, fileList in os.walk(testmrpath):
    for filename in fileList:
        filetoadd=255*np.array(nib.load(os.path.join(testmrpath,filename)).get_fdata())
        filetoadd=filetoadd[2:178,12:204,2:178]
        filetoadd=nib.Nifti1Image(filetoadd, np.eye(4))
        nib.save(filetoadd, "/hpc/jmcn735/Unet/CropFinal2/test/MR/"+filename)
        