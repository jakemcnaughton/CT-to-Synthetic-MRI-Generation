import tensorflow as tf 
import matplotlib.pyplot as plt
import nibabel as nib
import scipy
import os
#from fsl.wrappers import flirt, LOAD, applyxfm, fnirt, bet
import numpy as np
#from HD_BET.run import run_hd_bet
import torch
import torchio as tio


SPLIT=0.8
#gpus = tf.config.list_physical_devices('GPU')
#tf.config.set_visible_devices(gpus[0], 'GPU')

patients=[]
mris=[]
cts=[]

ctregistereddir="/hpc/jmcn735/RegCT"
mriregistereddir="/hpc/jmcn735/RMR"
ctdir="/hpc/jmcn735/CT"
mridir="/hpc/jmcn735/MR"
ct="/hpc/jmcn735/ECT"
mr="/hpc/jmcn735/EMR"
atlas="/hpc/jmcn735/Data/icbm_avg_152_t1_tal_lin.nii"


#mr="/hpc/jmcn735/Unet/15JanData/MRI"
#ct="/hpc/jmcn735/Unet/15JanData/CT"

trainmri="/hpc/jmcn735/Unet/Final/train/MR"
trainct="/hpc/jmcn735/Unet/Final/train/CT"

valmr = "/hpc/jmcn735/Unet/Final/val/MR"
valct = "/hpc/jmcn735/Unet/Final/val/CT"

testmr="/hpc/jmcn735/Unet/Final/test/MR"
testct="/hpc/jmcn735/Unet/Final/test/CT"


for root, dirs, files in os.walk(mr):
  for name in files:
      if name not in mris:
        mris.append(name[0:4])

for root, dirs, files in os.walk(ct):
  for name in files:
      cts.append(name[0:4])
print(len(cts))

for i in range(len(cts)-1):
  patient=cts[i]
  print(patient)

  
  if i < 144: 
    #ct = tio.ScalarImage(os.path.join(ct,patient+".nii.gz"))
    #mri = tio.ScalarImage(os.path.join(mr,patient+".nii.gz"))
    #transform=tio.CropOrPad((256,256,256))
    #paddedct=transform(ct)
    #paddedmri=transform(mri)
    #paddedct=nib.Nifti1Image(paddedct,np.eye(4))
    #paddedmri=nib.Nifti1Image(paddedmri,np.eye(4))
    #nib.save(paddedct, "/hpc/jmcn735/Unet/16Data256/train/CT/" +patient +".nii")
    #nib.save(paddedmri, "/hpc/jmcn735/Unet/16Data256/train/MR/" +patient + ".nii")
    os.rename(os.path.join(ct,patient+".nii.gz"), os.path.join(trainct,patient+".nii")) 
    os.rename(os.path.join(mr,patient+".nii.gz"), os.path.join(trainmri,patient+".nii"))
  #  print(i)
  elif i < 162:
    os.rename(os.path.join(ct,patient+".nii.gz"), os.path.join(valct,patient+".nii"))
    os.rename(os.path.join(mr,patient+".nii.gz"), os.path.join(valmr,patient+".nii"))
    #ct.save(os.path.join(valct,patient+".nii"))  
    #mri.save(os.path.join(valmri,patient+".nii"))
    #print(i)
  else:
    os.rename(os.path.join(ct,patient+".nii.gz"), os.path.join(testct,patient+".nii"))
    os.rename(os.path.join(mr,patient+".nii.gz"), os.path.join(testmr,patient+".nii"))
    #ct=nib.Nifti1Image(ct,np.eye(4))
    #mri=nib.Nifti1Image(mri,np.eye(4)) 
    #ct.save(os.path.join(testct,patient+".nii")) 
    #mri.save(os.path.join(testmri,patient+".nii"))"""
     
for mri in mris:
  if mri in cts:
      patients.append(mri)
  else:
      print(mri)

print(patients)
def NormalizeData(data):
    data=data.get_fdata()
    return nib.Nifti1Image(data/np.max(data), np.eye(4))


"""

#CT

#mri=nib.load(os.path.join(mridir,patient+"_T1.nii.gz"))

ct=nib.load(os.path.join(ctdir,patient+".nii.gz"))
#ct= nib.Nifti1Image(ct.get_fdata(), np.eye(4))
#ct = nib.save(ct, os.path.join(ctdir,patient+".nii.gz"))
#ct=nib.load(os.path.join(ctdir,patient+".nii.gz"))
#bet(mri, os.path.join("/hpc/jmcn735/extrMRinitial",patient+".nii.gz"))

#MRI
mri=nib.load(os.path.join(mridir,patient+"_T1.nii.gz"))
#mri= nib.Nifti1Image(mri.get_fdata(), np.eye(4))
#mri = nib.save(mri, os.path.join(mridir))
#mri=nib.load(os.path.join(mridir,patient+"_T1.nii.gz"))
#ALIGN CT TO MRI
#ctaligned=flirt(ct,mri,out=os.path.join(ctregistereddir,patient+"_mri_mi12"),cost="mutualinfo", dof=12)
#ctaligned=flirt(ct,mri,out=os.path.join(ctregistereddir,patient+"_mri_mi9"),cost="mutualinfo", dof=9)
#ctaligned=flirt(ct,mri,out=os.path.join(ctregistereddir,patient+"_mri_mi7"),cost="mutualinfo", dof=7)
#ctaligned=flirt(ct,mri,out=os.path.join(ctregistereddir,patient+"_mri_mi6"),cost="mutualinfo", dof=6)
#ctaligned=flirt(ct,mri,out=os.path.join(ctregistereddir,patient+"_mri_cr12"),cost="corratio", dof=12)
#ctaligned=flirt(ct,mri,out=os.path.join(ctregistereddir,patient+"_mri_cr9"),cost="corratio", dof=9)
#ctaligned=flirt(ct,mri,out=os.path.join(ctregistereddir,patient+"_mri_cr7"),cost="corratio", dof=7)
#ctaligned=flirt(ct,mri,out=os.path.join(ctregistereddir,patient+"_mri_cr6"),cost="corratio", dof=6)
mri=nib.load(os.path.join(mridir,patient+"_T1.nii.gz"))

ctaligned=nib.load(os.path.join(ctregistereddir,patient+"_mri.nii.gz"))
#ALIGNED MRI TO ATLAS
mrialigned=flirt(mri,atlas,out=os.path.join(mriregistereddir,patient),dof=12, omat="transform.mat")
#mrialigned=nib.load(os.path.join(mriregistereddir,patient+".nii.gz"))
#ALIGNED CT TO ATLAS
ctalignedtoatlas = applyxfm(src=ctaligned, ref=atlas,out=os.path.join(ctregistereddir,patient+"_atlas"), mat="transform.mat")
#ctaligned=nib.load(os.path.join(ctregistereddir,patient+"_atlas.nii.gz"))
#finalalignedct=flirt(ctaligned,ref=mrialigned,out="finalct",dof=6,cost='mutualinfo')



#EXTRACTED MRI AND MASK
run_hd_bet(os.path.join(mriregistereddir,patient), os.path.join(mriextrdir,patient+".nii.gz"))
ctdata=nib.load(os.path.join(ctregistereddir, patient+"_atlas.nii.gz")).get_fdata()
#MRI MASK
extr=nib.load(os.path.join(mriextrdir,patient+".ni_mask.nii.gz"))
mask=extr.get_fdata()
#EXTRACTED CT
stripped_ct = nib.Nifti1Image(mask*ctdata, np.eye(4))
nib.save(stripped_ct,os.path.join(ctextrdir,patient+".nii.gz"))

#DOWNSAMPLED CT
nmri=NormalizeData(nib.load(os.path.join(mriextrdir,patient+".nii.gz")))
nib.save(nmri,os.path.join(mriextrdir,patient+".nii.gz"))

filetoadd = tio.ScalarImage(os.path.join(ctextrdir,patient+".nii.gz"))
transform=tio.CropOrPad((210,210,210))
padded=transform(filetoadd)  
transform = tio.Resample(2)
ct=transform(padded)
transform=tio.CropOrPad((128,128,128)) 
ct = transform(ct)
"""
#DOWNSAMPLED MRI
#filetoadd = tio.ScalarImage(os.path.join(mriextrdir,patient+".nii.gz"))
#transform=tio.CropOrPad((210,210,210))
#padded=transform(filetoadd)  
#transform = tio.Resample(2)
#mri=transform(padded) 
#transform=tio.CropOrPad((128,128,128)) 
#mri = transform(mri)
#run_hd_bet(os.path.join(mriregistereddir,patient), os.path.join(mriextrdir,patient+".nii.gz"))
#ctdata=nib.load(os.path.join(ctregistereddir, patient+"_atlas.nii.gz")).get_fdata()
#MRI MASK

"""
extr=nib.load(os.path.join(mriextrdir,patient+".ni_mask.nii.gz"))
mask=extr.get_fdata()
#EXTRACTED CT
#stripped_ct = nib.Nifti1Image(mask*ctdata, np.eye(4))
#nib.save(stripped_ct,os.path.join(ctextrdir,patient+".nii.gz"))

#DOWNSAMPLED CT
nmri=NormalizeData(nib.load(os.path.join(mriextrdir,patient+".nii.gz")))
nib.save(nmri,os.path.join(mriextrdir,patient+".nii.gz"))

filetoadd = tio.ScalarImage(os.path.join(ctextrdir,patient+".nii.gz"))
transform=tio.CropOrPad((256,256,256))
padded=transform(filetoadd)  
#transform = tio.Resample(2)
#ct=transform(padded)
#transform=tio.CropOrPad((128,128,128)) 
#ct = transform(ct)


  #ct=nib.Nifti1Image(ct,np.eye(4))
"""