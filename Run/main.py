from UNet2D import *
import nibabel as nib
import numpy as np
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.callbacks import CSVLogger


BATCH_SIZE=4
EPOCHS=300
SAVE_FREQ=1
NAME="2D_b16_lr_5e5"
test = True
if test is True:
  gpus = tf.config.list_physical_devices('GPU')
  tf.config.set_visible_devices(gpus[0], 'GPU')
  
  testctpath="/hpc/jmcn735/Unet/Final/test/CT"
  testmrpath="/hpc/jmcn735/Unet/Final/test/MR"
  filelist=[]
  model = unet()
  model.load_weights('{}.hdf5'.format(NAME))
  
  
  testx=[]
  testy=[]
  for dirName, subdirList, fileList in os.walk(testctpath):
      for filename in fileList:
          filetoadd=np.array(nib.load(os.path.join(testctpath,filename)).get_fdata())
          filetoadd=filetoadd[:,12:204,:]
          filetoadd=np.pad(filetoadd, ((5, 6),(0,0),(5, 6)))
          for i in range(192):
            #filetoadd[:,:,i].reshape([192,192,1])
            testx.append(filetoadd[:,:,i].reshape([192,192,1]))
  
  for dirName, subdirList, fileList in os.walk(testmrpath):
      for filename in fileList:
          filetoadd=255*np.array(nib.load(os.path.join(testmrpath,filename)).get_fdata())
          filetoadd=filetoadd[:,12:204,:]
          filetoadd=np.pad(filetoadd, ((5, 6),(0,0),(5, 6)))
          for i in range(192):
            testy.append(filetoadd[:,:,i].reshape([192,192,1]))
  
  ct=np.stack(testx, axis=0)
  mri=np.stack(testy, axis=0)
  model.evaluate(ct, mri, batch_size=1)  
  
  
  for dirName, subdirList, fileList in os.walk(testctpath):
      for filename in fileList:
          filelist=[]
          filetoadd=np.array(nib.load(os.path.join(testctpath,filename)).get_fdata())
          filetoadd=filetoadd[:,12:204,:]
          filetoadd=np.pad(filetoadd, ((5, 6),(0,0),(5, 6)))
          for i in range(192):
            filelist.append(np.squeeze(model.predict(filetoadd[:,:,i].reshape([1, 192,192,1]))))
          filetoadd=np.stack(filelist, axis=0)
          print(tf.shape(filetoadd))
          #results=np.squeeze(filetoadd)
          filetoadd=np.swapaxes(filetoadd, 0, 2)
          result=nib.Nifti1Image(filetoadd, affine=np.eye(4))
          nib.save(result,"/hpc/jmcn735/Unet/{}/{}".format(NAME, filename))
  
  
  
  
else:
  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
    model = unet()
    
          
  trainpath="/hpc/jmcn735/Unet/Final/train/CT"
  targetpath="/hpc/jmcn735/Unet/Final/train/MR"
  valctpath="/hpc/jmcn735/Unet/Final/val/CT"
  valmripath="/hpc/jmcn735/Unet/Final/val/MR"
   
  trainlist=[]
  targetlist=[]
  valctlist=[]
  valmrilist=[]
  
  for dirName, subdirList, fileList in os.walk(trainpath):
      for filename in fileList:
          filetoadd=np.array(nib.load(os.path.join(trainpath,filename)).get_fdata())
          filetoadd=filetoadd[:,12:204,:]
          filetoadd=np.pad(filetoadd, ((5, 6),(0,0),(5, 6)))
          for i in range(192):
            trainlist.append(filetoadd[:,:,i].reshape([192,192,1]))
  
          
  for dirName, subdirList, fileList in os.walk(targetpath):
      for filename in fileList:
          filetoadd=255*np.array(nib.load(os.path.join(targetpath,filename)).get_fdata())
          filetoadd=filetoadd[:,12:204,:]
          filetoadd=np.pad(filetoadd, ((5, 6),(0,0),(5, 6)))
          for i in range(192):
            targetlist.append(filetoadd[:,:,i].reshape([192,192,1]))
  
  for dirName, subdirList, fileList in os.walk(valctpath):
      for filename in fileList:
          filetoadd=np.array(nib.load(os.path.join(valctpath,filename)).get_fdata())
          filetoadd=filetoadd[:,12:204,:]
          filetoadd=np.pad(filetoadd, ((5, 6),(0,0),(5, 6)))
          for i in range(192):
            valctlist.append(filetoadd[:,:,i].reshape([192,192,1]))
  
  for dirName, subdirList, fileList in os.walk(valmripath):
      for filename in fileList:
          filetoadd=255*np.array(nib.load(os.path.join(valmripath,filename)).get_fdata())
          filetoadd=filetoadd[:,12:204,:]
          filetoadd=np.pad(filetoadd, ((5, 6),(0,0),(5, 6)))
          for i in range(192):
            valmrilist.append(filetoadd[:,:,i].reshape([192,192,1]))
  
  ct=np.stack(trainlist, axis=0)
  mri=np.stack(targetlist, axis=0)
  valctset=np.stack(valctlist, axis=0)
  valmriset=np.stack(valmrilist, axis=0)
  
  csv_logger = CSVLogger('log_{}.csv'.format(NAME), append=True, separator=';')
  model_checkpoint = ModelCheckpoint('{}.hdf5'.format(NAME), verbose=1,save_freq='epoch',save_weights_only=True)
  history = model.fit(ct, mri,epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[model_checkpoint, csv_logger],validation_data=(valctset,valmriset))
