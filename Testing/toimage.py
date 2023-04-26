import numpy as np
from PIL import Image
import os
import nibabel as nib            

def DICE_COE(mask1, mask2):
    intersect = np.sum(mask1*mask2)
    fsum = np.sum(mask1)
    ssum = np.sum(mask2)
    dice = (2 * intersect ) / (fsum + ssum)
    dice = np.mean(dice)
    #dice = round(dice, 5) # for easy reading
    return dice   
    
    
model = "112"    
real_mri = nib.load("/hpc/jmcn735/Z885Seg/Z885TrueSeg.nii").get_fdata()
fake_mri = nib.load(f"/hpc/jmcn735/Z885Seg/Z885{model}Seg.nii").get_fdata()
print(model)
print(DICE_COE(real_mri, fake_mri))
    
"""    
for dirName, subdirList, fileList in os.walk(path):
      for model in subdirList: 
        for filename in fileList:
          if "Z101" in filename:
            filetoadd=np.array(nib.load(path+filename).get_fdata())
            #sliced = filetoadd[:,:,96]
            #im = Image.fromarray(sliced)
            #im = im.convert("L")
            #im.save("/hpc/jmcn735/{}/{}.jpeg".format(Foldername, filename[0:4]))
"""