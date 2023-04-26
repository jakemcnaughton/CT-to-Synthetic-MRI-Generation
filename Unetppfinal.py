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

def SSIM(y_true, y_pred):
  return 1-tf.reduce_mean(tf.image.ssim(y_true, y_pred, 255))
  

def PSNR(y_true, y_pred):
    max_pixel = 255
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))
  

def unet(pretrained_weights = None,input_size = (96,96,96,1)):
    
    #n-4th
    inputs = Input(input_size)
    x00 = Conv3D(64, 3, padding = 'same', activation = 'relu')(inputs)
    x00 = Conv3D(64, 3, padding = 'same', activation = 'relu')(x00)
    
    #n-3rd
    x10 = MaxPooling3D(pool_size=(2, 2, 2))(x00)
    x10 = Conv3D(128, 3, padding = 'same', activation = 'relu')(x10)
    x10 = Conv3D(128, 3, padding = 'same', activation = 'relu')(x10)
    upx10 = Conv3DTranspose(64,2,strides=(2,2,2),padding='same')(x10)
    
    #Skip connection
    x01 = concatenate([upx10, x00], axis=4)
    x01 = Conv3D(64, 3, padding = 'same', activation = 'relu')(x01)
    x01 = Conv3D(64, 3, padding = 'same', activation = 'relu')(x01)
    
    #n-2nd
    x20 = MaxPooling3D(pool_size=(2, 2, 2))(x10)
    x20 = Conv3D(256, 3, padding = 'same', activation = 'relu')(x20)
    x20 = Conv3D(256, 3, padding = 'same', activation = 'relu')(x20)
    upx20 = Conv3DTranspose(128,2,strides=(2,2,2),padding='same')(x20)
    
    #Skip connection
    x11 = concatenate([upx20, x10], axis=4)
    x11 = Conv3D(128, 3, padding = 'same', activation = 'relu')(x11)
    x11 = Conv3D(128, 3, padding = 'same', activation = 'relu')(x11)
    upx11 = Conv3DTranspose(64,2,strides=(2,2,2),padding='same')(x11)
    
    #Skip connection
    x02 = concatenate([upx11, x01,x00], axis=4)
    x02 = Conv3D(64, 3, padding = 'same', activation = 'relu')(x02)
    x02 = Conv3D(64, 3, padding = 'same', activation = 'relu')(x02)
    
    #n-1th
    x30 = MaxPooling3D(pool_size=(2, 2, 2))(x20)
    x30 = Conv3D(512, 3, padding = 'same', activation = 'relu')(x30)
    x30 = Conv3D(512, 3, padding = 'same', activation = 'relu')(x30)
    upx30 = Conv3DTranspose(256,2,strides=(2,2,2),padding='same')(x30)
    
    #Skip connection
    x21 = concatenate([upx30, x20],axis=4)
    x21 = Conv3D(256, 3, padding = 'same', activation = 'relu')(x21)
    x21 = Conv3D(256, 3, padding = 'same', activation = 'relu')(x21)
    upx21 = Conv3DTranspose(128,2,strides=(2,2,2),padding='same')(x21)
    
    #Skip connection
    x12 = concatenate([upx21, x11,x10], axis = 4)
    x12 = Conv3D(128, 3, padding = 'same', activation = 'relu',)(x12)
    x12 = Conv3D(128, 3, padding = 'same', activation = 'relu',)(x12)
    upx12 = Conv3DTranspose(64,2,strides=(2,2,2),padding='same')(x12)
    
    #Skip connection
    x03 = concatenate([upx12, x02,x01,x00], axis = 4)
    x03 = Conv3D(64, 3, padding = 'same', activation = 'relu',)(x03)
    x03 = Conv3D(64, 3, padding = 'same', activation = 'relu',)(x03)
    
    #BOTTOM
    x40 = MaxPooling3D(pool_size=(2, 2, 2))(x30)
    x40 = Conv3D(1024, 3, padding = 'same', activation = 'relu')(x40)
    x40 = Conv3D(1024, 3, padding = 'same', activation = 'relu')(x40)
    
    #n-1th
    x31 = Conv3DTranspose(512,2,strides=(2,2,2), padding='same')(x40)
    x31 = concatenate([x30,x31],axis=4)
    x31 = Conv3D(512, 3, padding = 'same', activation = 'relu')(x31)
    x31 = Conv3D(512, 3, padding = 'same', activation = 'relu')(x31)
    
    #n-2nd
    x22 = Conv3DTranspose(256,2,strides=(2,2,2),padding='same')(x31)
    x22 = concatenate([x22,x21,x20],axis=4)
    x22 = Conv3D(256, 3, padding = 'same', activation = 'relu')(x22)
    x22 = Conv3D(256, 3, padding = 'same', activation = 'relu')(x22)
    
    #n-3rd
    x13 = Conv3DTranspose(128,2,strides=(2,2,2),padding='same')(x22)
    x13 = concatenate([x13,x10,x11,x12],axis=4)
    x13 = Conv3D(128, 3, padding = 'same', activation = 'relu')(x13)
    x13 = Conv3D(128, 3, padding = 'same', activation = 'relu')(x13)
    
    #n-4th
    x04 = Conv3DTranspose(64,2,strides=(2,2,2),padding='same')(x13)
    x04 = concatenate([x04,x00,x01,x02,x03],axis=4)
    x04 = Conv3D(64, 3, padding = 'same', activation = 'relu')(x04)
    x04 = Conv3D(64, 3, padding = 'same', activation = 'relu')(x04)
    conv10 = Conv3D(1, 1, activation = 'relu')(x04)

    model = Model(inputs, conv10)
    
    
    
    model.compile(optimizer = Adam(learning_rate = 5e-5), loss = 'mean_absolute_error', metrics = ['MeanSquaredError',PSNR, SSIM])
    
    model.summary(line_length=160)

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model