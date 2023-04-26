#from Unet import *
from TransUnet import *
#from attentionUnet import *
#from UnetV2 import *
from keras.callbacks import CSVLogger
import nibabel as nib
import numpy as np
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify

BATCH_SIZE=1
EPOCHS=300
SAVE_FREQ=1
NAME = '300TransUnet1e6b1fullsize'


test = False
if test is True:
  gpus = tf.config.list_physical_devices('GPU')
  tf.config.set_visible_devices(gpus[0], 'GPU')
  testctpath = "/hpc/jmcn735/Unet/Final/test/CT"
  testmrpath = "/hpc/jmcn735/Unet/Final/test/MR" 
  model = unet()
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1, write_graph=True, write_images=True)
  
  model.load_weights('{}.hdf5'.format(NAME))
  testx=[]
  testy=[]
  for dirName, subdirList, fileList in os.walk(testctpath):
      for filename in fileList:
          filetoadd=np.array(nib.load(os.path.join(testctpath,filename)).get_fdata())
          filetoadd=filetoadd[:,12:204,:]
          filetoadd=np.pad(filetoadd, ((5, 6),(0,0),(5, 6)))
          testx.append(filetoadd.reshape([1,192,192,192,1]))
          
  for dirName, subdirList, fileList in os.walk(testmrpath):
      for filename in fileList:
          filetoadd=255*np.array(nib.load(os.path.join(testmrpath,filename)).get_fdata())
          filetoadd=filetoadd[:,12:204,:]
          filetoadd=np.pad(filetoadd, ((5, 6),(0,0),(5, 6)))
          testy.append(filetoadd.reshape([1,192,192,192,1]))
                    
  ct=np.concatenate(testx, axis=0)
  mri=np.concatenate(testy, axis=0)
  model.evaluate(ct, mri, callbacks=[tensorboard_callback], batch_size=1)    
      
      
  for dirName, subdirList, fileList in os.walk(testctpath):
    for filename in fileList:
        #Load File
        output_patches = np.empty([2,2,2,96,96,96])
        print(filename)
        filetoadd=np.array(nib.load(os.path.join(testctpath,filename)).get_fdata())
        filetoadd=filetoadd[:,12:204,:]
        filetoadd=np.pad(filetoadd, ((5, 6),(0,0),(5, 6)))
        for i in range(2):
          for j in range(2):
            for k in range(2):
              print([i, j, k])
              output_patches[i,j,k]=np.squeeze(model.predict(patchify(filetoadd, (96,96,96), step=96)[i,j,k].reshape([1,96,96,96,1])))
        
           
        #Patch it
        result=unpatchify(output_patches, [192, 192, 192])
        result=nib.Nifti1Image(result, affine=np.eye(4))
        nib.save(result,"/hpc/jmcn735/Unet/{}/".format(NAME)+filename[0:4]+".nii")
        
  
  
else:
  #THIS LETS IT USE MULTIPLE GPUS
  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
    model = unet()


  #config = tf.ConfigProto()
  #config.gpu_options.allow_growth = True
  #sess = tf.Session(config=config)
  try:
    os.mkdir("/hpc/jmcn735/Unet/{}".format(NAME))
  except FileExistsError:
    print('Directory not created.')

  try:
    os.mkdir("/hpc/jmcn735/Unet/logs/{}".format(NAME))
  except FileExistsError:
    print('Directory not created.')    
    
         
  trainpath="/hpc/jmcn735/Unet/Final/train/CT"
  targetpath="/hpc/jmcn735/Unet/Final/train/MR"
  valpath1="/hpc/jmcn735/Unet/Final/val/CT"
  valpath2="/hpc/jmcn735/Unet/Final/val/MR"
  
  
  trainx=[]
  trainy=[]
  valx=[]
  valy=[]
  patientsmr=[]
  patientsct=[]
  #181 x 217 x 181
  #181 x 192 x 181
  #192 x 192 x 192
  #THIS IS WHAT YOU DO IF YOU DON'T WANT TO USE THE PATCHING
  for dirName, subdirList, fileList in os.walk(trainpath):
      for filename in fileList:
          filetoadd=nib.load(os.path.join(trainpath,filename)).get_fdata()
          filetoadd=filetoadd[:,12:204,:]
          filetoadd=np.pad(filetoadd, ((5, 6),(0,0),(5, 6)))
          trainx.append(filetoadd.reshape([1,192,192,192,1]))    
          
  for dirName, subdirList, fileList in os.walk(targetpath):
      for filename in fileList:
          filetoadd=255*np.array(nib.load(os.path.join(targetpath,filename)).get_fdata())
          filetoadd=filetoadd[:,12:204,:]
          filetoadd=np.pad(filetoadd, ((5, 6),(0,0),(5, 6)))
          trainy.append(filetoadd.reshape([1,192,192,192,1]))
                    
  
  for dirName, subdirList, fileList in os.walk(valpath1):
      for filename in fileList:
          filetoadd=np.array(nib.load(os.path.join(valpath1,filename)).get_fdata())
          filetoadd=filetoadd[:,12:204,:]
          filetoadd=np.pad(filetoadd, ((5, 6),(0,0),(5, 6)))
          valx.append(filetoadd.reshape([1,192,192,192,1]))
          
          
  for dirName, subdirList, fileList in os.walk(valpath2):
      for filename in fileList:
          filetoadd=255*np.array(nib.load(os.path.join(valpath2,filename)).get_fdata())
          filetoadd=filetoadd[:,12:204,:]
          filetoadd=np.pad(filetoadd, ((5, 6),(0,0),(5, 6)))
          valy.append(filetoadd.reshape([1,192,192,192,1]))
            
  ct=np.concatenate(trainx, axis=0)
  mri=np.concatenate(trainy, axis=0)
  valct=np.concatenate(valx, axis=0)
  valmri=np.concatenate(valy, axis=0)
  
  csv_logger = CSVLogger('log_{}.csv'.format(NAME), append=True, separator=';')
  #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/{}".format(NAME), histogram_freq=1, write_graph=True, write_images=True)
  model_checkpoint = ModelCheckpoint('{}.hdf5'.format(NAME), verbose=1,save_freq='epoch',save_weights_only=True)
  history = model.fit(ct, mri,epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[model_checkpoint, csv_logger],validation_data=(valct,valmri))
  numpy_loss_history = np.array(history)
  np.savetxt("loss_history_{}.txt".format(NAME), numpy_loss_history, delimiter=",")
  
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model Loss (L1)')
  plt.ylim(0,0.075)
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig('/hpc/jmcn735/Unet/{}/Loss.png'.format(NAME))
  plt.close()

  plt.plot(history.history['PSNR'])
  plt.plot(history.history['val_PSNR'])
  plt.title('PSNR')
  plt.ylim(0,40)
  plt.ylabel('PSNR')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig('/hpc/jmcn735/Unet/{}/PSNR.png'.format(NAME))
  plt.close()
  
  plt.plot(history.history['SSIM'])
  plt.plot(history.history['val_SSIM'])
  plt.title('model SSIM')
  plt.ylabel('1-SSIM')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig('/hpc/jmcn735/Unet/{}/SSIM.png'.format(NAME))
  plt.close()
  
  plt.plot(history.history['mean_squared_error'])
  plt.plot(history.history['val_mean_squared_error'])
  plt.title('MSE')
  plt.ylim(0,0.02)
  plt.ylabel('MSE')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig('/hpc/jmcn735/Unet/{}/MSE.png'.format(NAME))