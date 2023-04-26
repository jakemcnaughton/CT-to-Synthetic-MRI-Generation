import numpy as np 
import os
import numpy as np
import keras.backend as K
from keras.models import *
from keras.layers import *
from tensorflow.keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import tensorflow as tf
import math


from keras.losses import mean_squared_error


def SSIM(y_true, y_pred):
  return 1-tf.reduce_mean(tf.image.ssim(y_true, y_pred, 255))
  

def PSNR(y_true, y_pred):
    max_pixel = 255
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))


def sqrtloss(y_true, y_pred):
  return K.mean(K.sqrt(K.abs(y_true-y_pred)))**2


  
def L1L2(a,b):
  return tf.keras.metrics.mean_absolute_error(a,b)+tf.keras.metrics.mean_squared_error(a,b)
  
def L1L2PSNR(a,b):
  return (tf.keras.metrics.mean_absolute_error(a,b)+tf.keras.metrics.mean_squared_error(a,b))/abs(tf.image.psnr(a,b,1))
  

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(mean_squared_error(y_true, y_pred))

def unet(pretrained_weights = None,input_size = (96,96,96,1)):
    inputs = Input(input_size) 
    conv1 = Conv3D(64, 3, activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv3D(64, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(128, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv3D(128, 3, activation = 'relu', padding = 'same' )(conv2)
    pool1 = MaxPooling3D(pool_size=(2, 2,2))(conv2)
   
    conv3 = Conv3D(256, 3, activation = 'relu', padding = 'same' )(pool1)
    conv3 = Conv3D(256, 3, activation = 'relu', padding = 'same' )(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2,2))(conv3)

    conv4 = Conv3D(512, 3, activation = 'relu', padding = 'same' )(pool3)
    conv4 = Conv3D(512, 3, activation = 'relu', padding = 'same' )(conv4)
    pool4 = MaxPooling3D(pool_size=(2,2,2))(conv4)
    
    conv5 = Conv3D(1024, 3, activation = 'relu', padding = 'same' )(pool4)
    conv5 = Conv3D(1024, 3, activation = 'relu', padding = 'same' )(conv5)

    up6= Conv3DTranspose(512,2,strides=(2,2,2),padding='same')(conv5)
    merge6 = concatenate([conv4,up6], axis = 4)
    conv6 = Conv3D(512, 3, activation = 'relu', padding = 'same' )(merge6)
    conv6 = Conv3D(512, 3, activation = 'relu', padding = 'same' )(conv6)

    up7= Conv3DTranspose(256,2,strides=(2,2,2),padding='same')(conv6)
    merge7 = concatenate([conv3,up7], axis = 4)
    conv7 = Conv3D(256, 3, activation = 'relu', padding = 'same' )(merge7)
    conv7 = Conv3D(256, 3, activation = 'relu', padding = 'same')(conv7)

    up8= Conv3DTranspose(128,2,strides=(2,2,2),padding='same')(conv7)
    merge8 = concatenate([conv2,up8], axis = 4)
    conv8 = Conv3D(128, 3, activation = 'relu', padding = 'same')(merge8)
    conv8 = Conv3D(128, 3, activation = 'relu', padding = 'same')(conv8)

    up9= Conv3DTranspose(64,2,strides=(2,2,2),padding='same')(conv8)
    merge9 = concatenate([conv1,up9], axis = 4)
    conv9 = Conv3D(64, 3, activation = 'relu', padding = 'same')(merge9)
    conv9 = Conv3D(64, 3, activation = 'relu', padding = 'same')(conv9)
    conv10 = Conv3D(1, 1, activation = 'relu')(conv9)

    model = Model(inputs, conv10)
    
    
    model.compile(optimizer = Adam(learning_rate = 5e-5), loss = 'MeanAbsoluteError', metrics = ['MeanSquaredError',PSNR, SSIM])
    
    model.summary(line_length=160)

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model