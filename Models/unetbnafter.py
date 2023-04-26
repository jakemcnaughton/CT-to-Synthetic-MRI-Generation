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


  
def L2L2(a,b):
  return tf.keras.metrics.mean_absolute_error(a,b)+tf.keras.metrics.mean_squared_error(a,b)
  
def L2L2PSNR(a,b):
  return (tf.keras.metrics.mean_absolute_error(a,b)+tf.keras.metrics.mean_squared_error(a,b))/abs(tf.image.psnr(a,b,1))
  

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(mean_squared_error(y_true, y_pred))

def unet(pretrained_weights = None,input_size = (176,192,176,1)):
    inputs = Input(input_size) 
    conv1 = Conv3D(64, 3, padding = 'same'  )(inputs)
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(64, 3,  padding = 'same' )(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(128, 3,  padding = 'same' )(pool2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv3D(128, 3,  padding = 'same' )(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2,2))(conv2)
   
    conv3 = Conv3D(256, 3,  padding = 'same' )(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv3D(256, 3,  padding = 'same' )(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2,2))(conv3)

    conv4 = Conv3D(512, 3,  padding = 'same' )(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv3D(512, 3,  padding = 'same' )(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling3D(pool_size=(2,2,2))(conv4)
    
    conv5 = Conv3D(1024, 3,  padding = 'same' )(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv3D(1024, 3,  padding = 'same' )(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    #conv5 = Conv3D(1024, 3,  padding = 'same' )(conv5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Activation('relu')(conv5)
    
    up6= Conv3DTranspose(512,2,strides=(2,2,2),padding='same',kernel_regularizer=tf.keras.regularizers.L2(l2=1e-5),use_bias=True)(conv5)
    #up6 = UpSampling3D(size=2)(conv5)
    #conv6 = Conv3D(1024, 3,  padding = 'same' )(up6)
    merge6 = concatenate([conv4,up6], axis = 4)
    conv6 = Conv3D(512, 3,  padding = 'same' )(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv3D(512, 3,  padding = 'same' )(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    
    up7= Conv3DTranspose(256,2,strides=(2,2,2),padding='same',kernel_regularizer=tf.keras.regularizers.L2(l2=1e-5),use_bias=True)(conv6)
    #up7 = UpSampling3D(size=2)(conv6)
    #conv7 = Conv3D(512, 3,  padding = 'same' )(up7)
    merge7 = concatenate([conv3,up7], axis = 4)
    conv7 = Conv3D(256, 3,  padding = 'same' )(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv3D(256, 3,  padding = 'same' )(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    
    up8= Conv3DTranspose(128,2,strides=(2,2,2),padding='same',kernel_regularizer=tf.keras.regularizers.L2(l2=1e-5),use_bias=True)(conv7)
    #up8 = UpSampling3D(size=2)(conv7)
    #conv8 = Conv3D(256, 3,  padding = 'same' )(up8)
    merge8 = concatenate([conv2,up8], axis = 4)
    conv8 = Conv3D(128, 3,  padding = 'same' )(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv3D(128, 3,  padding = 'same' )(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    #up9 = UpSampling3D(size=2)(conv8)
    #conv9 = Conv3D(128, 3,  padding = 'same' )(up9)
    up9= Conv3DTranspose(64,2,strides=(2,2,2),padding='same',kernel_regularizer=tf.keras.regularizers.L2(l2=1e-5),use_bias=True)(conv8)
    merge9 = concatenate([conv1,up9], axis = 4)
    conv9 = Conv3D(64, 3,  padding = 'same' )(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv3D(64, 3,  padding = 'same' )(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv10 = Conv3D(1, 1, activation = 'relu')(conv9)

    model = Model(inputs, conv10)
    
    
    model.compile(optimizer = Adam(learning_rate = 5e-4), loss = 'MeanAbsoluteError', metrics = ['MeanSquaredError',PSNR, SSIM])
    
    model.summary(line_length=160)

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model