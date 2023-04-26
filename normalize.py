import numpy as np
import nibabel as nib
import os
import torchio as tio

mridir="/hpc/jmcn735/Data/ExtrMRI"
ctdir="/hpc/jmcn735/Data/ExtrCT"
ctnormdir="/hpc/jmcn735/Data/CTNorm"
mrinormdir="/hpc/jmcn735/Data/MRINorm"

def NormalizeData(data):
    data=data.get_fdata()
    return nib.Nifti1Image(data/np.max(data), np.eye(4))
    
for root, dirs, files in os.walk(mridir):
    for i in range(int(len(files)/2)):
      patient=files[2*i][0:4]
      mri=nib.load(os.path.join(mridir,patient)+".nii.gz")
      nib.save(NormalizeData(mri),os.path.join(mrinormdir,patient))
      if i <120:
        filetoadd = tio.ScalarImage(os.path.join(mrinormdir,patient+".nii"))
        transform=tio.CropOrPad((192,192,192))
        padded=transform(filetoadd)  
        padded.save(os.path.join("/hpc/jmcn735/Unet/Data1/train/NMRI",patient+".nii"))
        print("{} saved".format(patient))
      else:
        patient=files[2*i]
        patient=patient[0:4]
        filetoadd = tio.ScalarImage(os.path.join(mrinormdir,patient+".nii"))
        transform=tio.CropOrPad((192,192,192))
        padded=transform(filetoadd)  
        padded.save(os.path.join("/hpc/jmcn735/Unet/Data1/test/NMRI",patient+".nii"))
        print("{} saved".format(patient))
      
     
      
for root, dirs, files in os.walk(ctdir):
    for j in range(int(len(files))):
      patient=files[j][0:4]
      ct=nib.load(os.path.join(ctdir,patient)+".nii.gz")
      data=ct.get_fdata()
      nib.save(nib.Nifti1Image(data, np.eye(4)),os.path.join(ctnormdir,patient))
      if j <120:
        filetoadd = tio.ScalarImage(os.path.join(ctnormdir,patient+".nii"))
        transform=tio.CropOrPad((192,192,192))
        padded=transform(filetoadd)  
        padded.save(os.path.join("/hpc/jmcn735/Unet/Data1/train/CT",patient+".nii"))
        print("{} saved".format(patient))
      else:
        filetoadd = tio.ScalarImage(os.path.join(ctnormdir,patient+".nii"))
        transform=tio.CropOrPad((192,192,192))
        padded=transform(filetoadd)  
        padded.save(os.path.join("/hpc/jmcn735/Unet/Data1/test/CT",patient+".nii"))
        print("{} saved".format(patient))