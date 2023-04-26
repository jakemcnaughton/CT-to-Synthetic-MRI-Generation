import nibabel as nib
import numpy as np
import os
import torch
import torchio as tio


mris=[]
cts=[]


mridir="/hpc/jmcn735/Unet/Data/train/MRI"
ctdir="/hpc/jmcn735/Unet/Data/train/CT"

valctdir ="/hpc/jmcn735/Unet/Data/val/CT"
valmridir = "/hpc/jmcn735/Unet/Data/val/MRI"

testmridir="/hpc/jmcn735/Unet/Data/test/MRI"
testctdir="/hpc/jmcn735/Unet/Data/test/CT"


for root, dirs, files in os.walk(ctdir):
  for name in files:
      mris.append(name[0:4])


for root, dirs, files in os.walk(valmridir):
  for name in files:
      cts.append(name[0:4])
      
transform=tio.CropOrPad((96,96,96))
"""
for patient in mris:
  trainmri=tio.ScalarImage(os.path.join(mridir,patient+".nii"))
  trainct=tio.ScalarImage(os.path.join(ctdir,patient+".nii"))
  trainmri=transform(trainmri)
  trainct=transform(trainct)
  trainmri.save(os.path.join(mridir,patient+".nii"))
  trainct.save(os.path.join(ctdir,patient+".nii"))
  print(patient)
"""
for patient in cts:

  testct=tio.ScalarImage(os.path.join(valctdir,patient+".nii"))
  testmri=tio.ScalarImage(os.path.join(valmridir,patient+".nii"))
  testmri=transform(testmri)
  testct=transform(testct)
  testmri.save(os.path.join(valmridir,patient+".nii"))
  testct.save(os.path.join(valctdir,patient+".nii"))
  print(patient)
  
  