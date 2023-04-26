import os
import nibabel as nib
import numpy as np

mriextrdir="/hpc/jmcn735/3D-CycleGan-Pytorch-MedImaging/Data/train/CT"
newdir="/hpc/jmcn735/3D-CycleGan-Pytorch-MedImaging/Data/train255/CT"

patients=[]

for root, dirs, files in os.walk(mriextrdir):
  for name in files:
      patients.append(name)
      
      
def NormalizeData(data):
    data=data.get_fdata()
    return nib.Nifti1Image(data*255, np.eye(4))
    
for patient in patients:  
    nmri=nib.load(os.path.join(mriextrdir,patient))
    nib.save(nmri,os.path.join(newdir,patient))