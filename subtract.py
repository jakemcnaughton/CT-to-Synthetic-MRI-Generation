import os
import numpy as np
import nibabel as nib
NAME= 'PP300e4'
mripath = "/hpc/jmcn735/Unet/Data/test/MRI"
spath = "/hpc/jmcn735/Unet/{}".format(NAME)
diffpath = "/hpc/jmcn735/Unet/{}_diff".format(NAME)

try:
      os.mkdir("/hpc/jmcn735/Unet/{}_diff".format(NAME))
except FileExistsError:
      print('Directory not created.')
  
for dirName, subdirList, fileList in os.walk(mripath):
      for filename in fileList:
          mri=np.array(nib.load(os.path.join(mripath,filename)).get_fdata())
          synth=np.array(nib.load(os.path.join(spath,filename)).get_fdata())
          diff = nib.Nifti1Image(mri-synth, np.eye(4))
          nib.save(diff, os.path.join(diffpath,filename))
          print(filename)
      

  