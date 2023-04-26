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
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - 
y_true))))  
  
def convblock(filters, activation_func, input_name):
  x00 = Conv3D(filters, 3, padding = 'same')(input_name)
  #x00 = BatchNormalization()(x00)
  x00 = Activation('relu')(x00)
  x00 = Conv3D(filters, 3, padding = 'same')(x00)
  #x00 = BatchNormalization()(x00)
  return Activation('relu')(x00)
  
 
def attention_block(x, gating, inter_shape):


    theta_x = Conv3D(inter_shape, (1, 1, 1), strides=(2, 2, 2))(x)  
    phi_g = Conv3D(inter_shape, (1, 1, 1), padding='same')(gating)
    
    
    concat_xg = add([phi_g, theta_x])
    #ReLu -> Conv -> Sigmoid
    act_xg = Activation('relu')(concat_xg)
    psi = Conv3D(1, (1, 1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    #Resample
    upsample_psi = UpSampling3D(size=(2, 2, 2))(sigmoid_xg)
    #Multiply
    y = multiply([upsample_psi, x])

    result = Conv3D(inter_shape, (1, 1, 1), padding='same')(y)
    #result_bn = BatchNormalization()(result)
    return result


def unet(pretrained_weights = None,input_size = (96,96,96,1)):
    
    
    inputs = Input(input_size)
    x0d = convblock(64, 'relu', inputs)
    x1d = MaxPooling3D(pool_size=(2, 2, 2))(x0d)
    x1d = convblock(128, 'relu', x1d)
    x2d = MaxPooling3D(pool_size=(2, 2, 2))(x1d)
    x2d = convblock(256, 'relu', x2d)
    x3d = MaxPooling3D(pool_size=(2, 2, 2))(x2d)
    x3d = convblock(512, 'relu', x3d)
    x4b = MaxPooling3D(pool_size=(2, 2, 2))(x3d)
    x4b = convblock(1024, 'relu', x4b)
    
    x3u = Conv3DTranspose(512,2,strides=(2,2,2),padding='same')(x4b)
    att3 = attention_block(x3d, x4b, 512)
    x3u = concatenate([x3u, att3], axis=4)
    x3u = convblock(512, 'relu', x3u)
    
    x2u = Conv3DTranspose(256,2,strides=(2,2,2),padding='same')(x3u)
    att2 = attention_block(x2d, x3u, 256)
    x2u = concatenate([x2u, att2], axis=4)
    x2u = convblock(256, 'relu', x2u)
    
    x1u= Conv3DTranspose(128,2,strides=(2,2,2),padding='same')(x2u)
    att1 = attention_block(x1d, x2u, 128)
    x1u = concatenate([x1u, att1], axis=4)
    x1u = convblock(128, 'relu', x1u)
    
    x0u= Conv3DTranspose(64,2,strides=(2,2,2),padding='same')(x1u)
    att0 = attention_block(x0d, x1u, 64)
    x0u = concatenate([x0u, att0], axis=4)
    x0u = convblock(64, 'relu', x0u)
    
    conv10 = Conv3D(1, 1, activation = 'relu')(x0u)
    model = Model(inputs, conv10)    
    
    
    model.compile(optimizer = Adam(learning_rate = 5e-5), loss = 'mean_absolute_error', metrics = ['MeanSquaredError',PSNR, SSIM])
    
    model.summary(line_length=160)

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model