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
  return 1-tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
  

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))
  

def unet(pretrained_weights = None,input_size = (128,128,128,1)):
    
    #n-4th
    inputs = Input(input_size)
    x00 = Conv3D(64, 3, activation = 'relu', padding = 'same')(inputs)
    x00 = BatchNormalization()(x00)
    x00 = Conv3D(64, 3, activation = 'relu', padding = 'same')(x00)
    x00 = BatchNormalization()(x00)
    
    #n-3rd
    x10 = MaxPooling3D(pool_size=(2, 2, 2))(x00)
    x10 = Conv3D(128, 3, activation = 'relu', padding = 'same')(x10)
    x10 = BatchNormalization()(x10)
    x10 = Conv3D(128, 3, activation = 'relu', padding = 'same')(x10)
    x10 = BatchNormalization()(x10)
    upx10 = UpSampling3D(size=2)(x10)
    
    #Skip connection
    x01 = concatenate([upx10, x00], axis=4)
    x01 = Conv3D(64, 3, activation = 'relu', padding = 'same')(x01)
    x01 = BatchNormalization()(x01)
    x01 = Conv3D(64, 3, activation = 'relu', padding = 'same')(x01)
    x01 = BatchNormalization()(x01)
    
    #n-2nd
    x20 = MaxPooling3D(pool_size=(2, 2, 2))(x10)
    x20 = Conv3D(256, 3, activation = 'relu', padding = 'same')(x20)
    x20 = BatchNormalization()(x20)
    x20 = Conv3D(256, 3, activation = 'relu', padding = 'same')(x20)
    x20 = BatchNormalization()(x20)
    upx20 = UpSampling3D(size=2)(x20)
    
    #Skip connection
    x11 = concatenate([upx20, x10], axis=4)
    x11 = Conv3D(128, 3, activation = 'relu', padding = 'same')(x11)
    x11 = BatchNormalization()(x11)
    x11 = Conv3D(128, 3, activation = 'relu', padding = 'same')(x11)
    x11 = BatchNormalization()(x11)
    upx11 = UpSampling3D(size=2)(x11)
    
    #Skip connection
    x02 = concatenate([upx11, x01,x00], axis=4)
    x02 = Conv3D(64, 3, activation = 'relu', padding = 'same')(x02)
    x02 = BatchNormalization()(x02)
    x02 = Conv3D(64, 3, activation = 'relu', padding = 'same')(x02)
    x02 = BatchNormalization()(x02)
    
    
    #n-1th
    x30 = MaxPooling3D(pool_size=(2, 2, 2))(x20)
    x30 = Conv3D(512, 3, activation = 'relu', padding = 'same')(x30)
    x30 = BatchNormalization()(x30)
    x30 = Conv3D(512, 3, activation = 'relu', padding = 'same')(x30)
    x30 = BatchNormalization()(x30)
    upx30 = UpSampling3D(size=2)(x30)
    
    #Skip connection
    x21 = concatenate([upx30, x20],axis=4)
    x21 = Conv3D(256, 3, activation = 'relu', padding = 'same')(x21)
    x21 = BatchNormalization()(x21)
    x21 = Conv3D(256, 3, activation = 'relu', padding = 'same')(x21)
    x21 = BatchNormalization()(x21)
    upx21 = UpSampling3D(size=2)(x21)
    
    #Skip connection
    x12 = concatenate([upx21, x11,x10], axis = 4)
    x12 = Conv3D(128, 3, activation = 'relu', padding = 'same')(x12)
    x12 = BatchNormalization()(x12)
    x12 = Conv3D(128, 3, activation = 'relu', padding = 'same')(x12)
    x12 = BatchNormalization()(x12)
    upx12 = UpSampling3D(size=2)(x12)
    
    #Skip connection
    x03 = concatenate([upx12, x02,x01,x00], axis = 4)
    x03 = Conv3D(64, 3, activation = 'relu', padding = 'same')(x03)
    x03 = BatchNormalization()(x03)
    x03 = Conv3D(64, 3, activation = 'relu', padding = 'same')(x03)
    x03 = BatchNormalization()(x03)
    
    
    #BOTTOM
    x40 = MaxPooling3D(pool_size=(2, 2, 2))(x30)
    x40 = Conv3D(1024, 3, activation = 'relu', padding = 'same')(x40)
    x40 = BatchNormalization()(x40)
    x40 = Conv3D(1024, 3, activation = 'relu', padding = 'same')(x40)
    x40 = BatchNormalization()(x40)
    
    #n-1th
    x31 = UpSampling3D(size=2)(x40)
    x31 = concatenate([x30,x31],axis=4)
    x31 = Conv3D(512, 3, activation = 'relu', padding = 'same')(x31)
    x31 = BatchNormalization()(x31)
    x31 = Conv3D(512, 3, activation = 'relu', padding = 'same')(x31)
    x31 = BatchNormalization()(x31)
    
    #n-2nd
    x22 = UpSampling3D(size=2)(x31)
    x22 = concatenate([x22,x21,x20],axis=4)
    x22 = Conv3D(256, 3, activation = 'relu', padding = 'same')(x22)
    x22 = BatchNormalization()(x22)
    x22 = Conv3D(256, 3, activation = 'relu', padding = 'same')(x22)
    x22 = BatchNormalization()(x22)
    
    #n-3rd
    x13 = UpSampling3D(size=2)(x22)
    x13 = concatenate([x13,x10,x11,x12],axis=4)
    x13 = Conv3D(128, 3, activation = 'relu', padding = 'same')(x13)
    x13 = BatchNormalization()(x13)
    x13 = Conv3D(128, 3, activation = 'relu', padding = 'same')(x13)
    x13 = BatchNormalization()(x13)
    
    #n-4th
    x04 = UpSampling3D(size=2)(x13)
    x04 = concatenate([x04,x00,x01,x02,x03],axis=4)
    x04 = Conv3D(64, 3, activation = 'relu', padding = 'same')(x04)
    x04 = BatchNormalization()(x04)
    x04 = Conv3D(64, 3, activation = 'relu', padding = 'same')(x04)
    x04 = BatchNormalization()(x04)
    conv10 = Conv3D(1, 1, activation = 'relu')(x04)

    model = Model(inputs, conv10)
    
    
    
    model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'mean_absolute_error', metrics = ['MeanSquaredError',PSNR, SSIM])
    
    model.summary(line_length=160)

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model