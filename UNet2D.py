import os
import math
import numpy as np
import keras.backend as K
from keras.models import *
from keras.layers import *
from tensorflow.keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import tensorflow as tf


def SSIM(y_true, y_pred):
  return 1-tf.reduce_mean(tf.image.ssim(y_true, y_pred, 255.0))
  
#def PSNR(y_true, y_pred):
 #   max_pixel = 255
  #  print(K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true)))))
   # return 10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))  


def PSNR(original, contrast):
    mse=tf.keras.metrics.mean_squared_error(original, contrast, reduction="SUM")
    print(mse)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    PSNR = 10 * K.log(PIXEL_MAX**2 / mse)
    return PSNR
    
    
def unet(pretrained_weights = None,input_size = (192,192,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
   
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(conv4)
    
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(conv5)
    conv5 = BatchNormalization()(conv5)
    
    
    up66= Conv2DTranspose(512,2,strides=(2,2),padding='same')(conv5)
    merge66 = concatenate([conv4,up66], axis = 3)
    conv66 = Conv2D(512, 3, activation = 'relu', padding = 'same')(merge66)
    conv66 = BatchNormalization()(conv66)
    conv66 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv66)
    conv66 = BatchNormalization()(conv66)

    up7= Conv2DTranspose(256,2,strides=(2,2),padding='same')(conv66)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8= Conv2DTranspose(128,2,strides=(2,2),padding='same')(conv7)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9= Conv2DTranspose(64,2,strides=(2,2),padding='same')(conv8)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(1, 1, activation = 'relu')(conv9)

    model = Model(inputs, conv10)
    
    
    model.compile(optimizer = Adam(learning_rate = 5e-5), loss = 'mean_absolute_error', metrics = ['MeanSquaredError',SSIM])
    
    model.summary(line_length=160)

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model