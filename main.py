from model import *
from data import *
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
  

model = build_model()
#Save every epoch
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)

model.fit(trainlist,testlist,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])
#model.predict()
results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/membrane/test",results)
