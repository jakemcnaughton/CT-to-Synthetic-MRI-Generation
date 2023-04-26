import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import keras.backend as K
from keras.models import *
import nibabel as nib
from keras.layers import *
from tensorflow.keras.optimizers import *
#from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')
trainpath="/hpc/jmcn735/Data/ExtrMRI"

for z,y,filenames in os.walk(trainpath):

  for filename in filenames:
    #print(filename)
    if filename[0]=='A':
      print(filename)
      x=np.array(nib.load(os.path.join(trainpath,filename)).get_fdata())
      print(tf.shape(x))
      results=MaxPooling3D(pool_size=(2, 2,2))(x)
      result=nib.Nifti1Image(results, affine=np.eye(4))
      nib.save(result,os.path.join("/hpc/jmcn735/3D-CycleGan-Pytorch-MedImaging/Data/MRI", filename))
