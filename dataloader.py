import nibabel as nib
import numpy as np
import os
trainpath="/hpc/jmcn735/3D-CycleGan-Pytorch-MedImaging/Data_folder/train/images"
testpath=""
trainlist=[]
testlist=[]
for dirName, subdirList, fileList in os.walk(trainpath):
  for filename in fileList:
    trainlist.append(nib.load(os.path.join(trainpath,filename)).get_fdata())
    
for dirName, subdirList, fileList in os.walk(testpath):
  for filename in fileList:
    testlist.append(nib.load(os.path.join(testpath,filename)).get_fdata())
  
