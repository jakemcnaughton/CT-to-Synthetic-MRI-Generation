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
import tensorflow as tf
from tensorflow.keras import layers

def SSIM(y_true, y_pred):
  return 1-tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
  

def PSNR(y_true, y_pred):
    max_pixel = 255.0
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - 
y_true))))


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout_rate):
        super(TransformerBlock, self).__init__()

        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.feed_forward = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(embedding_dim)
        ])
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, mask=None):
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        #attention_output = self.dropout1(attention_output)
        skip_connection1 = concatenate([inputs, attention_output], axis=-1)

        feed_forward_output = self.feed_forward(skip_connection1)
        #feed_forward_output = self.dropout2(feed_forward_output)
        transformer_output = concatenate([skip_connection1, feed_forward_output], axis=-1)

        return transformer_output


class TransformerUNet(tf.keras.Model):
    def __init__(self, d_model=64, num_heads=4, ff_dim=128, rate=0):
        super(TransformerUNet, self).__init__()

        # Encoder part
        self.encoder1 = TransformerBlock(d_model, num_heads, ff_dim, rate)
        self.pool1 = layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))
        self.encoder2 = TransformerBlock(d_model*2, num_heads, ff_dim, rate)
        self.pool2 = layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))
        self.encoder3 = TransformerBlock(d_model * 4, num_heads, ff_dim, rate)
        self.pool3 = layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2),)
        self.encoder4 = TransformerBlock(d_model * 8, num_heads, ff_dim, rate)
        self.pool4 = layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))

        # Bottleneck part
        self.bottleneck = TransformerBlock(d_model * 16, num_heads, ff_dim, rate)

        # Decoder part
        self.upconv4 = layers.Conv3DTranspose(d_model * 8, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same')
        self.decoder4 = TransformerBlock(d_model * 8, num_heads, ff_dim, rate)
        self.upconv3 = layers.Conv3DTranspose(d_model * 4, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same')
        self.decoder3 = TransformerBlock(d_model * 4, num_heads, ff_dim, rate)
        self.upconv2 = layers.Conv3DTranspose(d_model * 2, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same')
        self.decoder2 = TransformerBlock(d_model * 2, num_heads, ff_dim, rate)
        self.upconv1 = layers.Conv3DTranspose(d_model, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same')
        self.decoder1 = TransformerBlock(d_model, num_heads, ff_dim, rate)

        # Output layer
        self.output_layer = layers.Conv3D(1, kernel_size=(1, 1, 1), activation='relu')

    def call(self, inputs):
        # Encoder part
        x = self.encoder1(inputs)
        x = self.pool1(x)
        x = self.encoder2(x)
        x = self.pool2(x)
        x = self.encoder3(x)
        x = self.pool3(x)
        x = self.encoder4(x)
        x = self.pool4(x)

        # Bottleneck part
        x = self.bottleneck(x)

        # Decoder
        x = self.upconv4(x)
        x = self.decoder4(x)
        x = self.upconv3(x)
        x = self.decoder3(x)
        x = self.upconv2(x)
        x = self.decoder2(x)
        x = self.upconv1(x)
        x = self.decoder1(x)
        x = self.output_layer(x)
        return x


def unet():
  model = TransformerUNet()
  model.build(input_shape=(1, 96,96,96, 1))
      
  model.compile(optimizer = Adam(learning_rate = 1e-6, clipnorm=1), loss = 'MeanAbsoluteError', metrics = ['MeanSquaredError',PSNR, SSIM])
  
  model.summary(line_length=160)
  
  #if(pretrained_weights):
  #    model.load_weights(pretrained_weights)
  
  return model