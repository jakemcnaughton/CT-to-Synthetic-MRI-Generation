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


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=4):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"Embedding dimension ({self.embed_dim}) must be divisible by the number of heads ({self.num_heads})")
        self.projection_dim = self.embed_dim // self.num_heads
        self.query_dense = layers.Dense(self.embed_dim)
        self.key_dense = layers.Dense(self.embed_dim)
        self.value_dense = layers.Dense(self.embed_dim)
        #self.combine_heads = layers.Dense(self.embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, x.shape[1], x.shape[2], x.shape[3], self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 4, 1, 2, 3, 5])
    
    def combine_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.transpose(x, perm=[0, 2, 3, 4, 1, 5])
        x = tf.reshape(x, (batch_size, x.shape[1], x.shape[2], x.shape[3], self.embed_dim))
        return x


    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        #attention = tf.transpose(attention, perm=[0, 2, 1, 2, 3, 5])
        #concat_attention = tf.reshape(attention, (batch_size, attention.shape[1], attention.shape[2], attention.shape[3], self.embed_dim))
        output = self.combine_heads(attention)
        return output, weights


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.conv = layers.Conv3D(embed_dim, (3, 3, 3), padding='same')

    def call(self, inputs, training):
        attn_output, _ = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        inputs = self.conv(inputs) 
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
        # Define the MultiHeadSelfAttention and TransformerBlock classes here (same as in the previous code)

class TransformerUNet(tf.keras.Model):
    def __init__(self, image_size, patch_size, d_model=128, num_heads=4, ff_dim=512, rate=0.1):
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
        self.x4_1 = Conv3D(512, 3, padding = 'same', activation='relu')
        self.x4_2 = Conv3D(512, 3, padding = 'same', activation='relu')
     
        
        #self.decoder4 = TransformerBlock(d_model * 8, num_heads, ff_dim, rate)
        self.upconv3 = layers.Conv3DTranspose(256, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same')
        self.x3_1 = Conv3D(256, 3, padding = 'same', activation='relu')
        self.x3_2 = Conv3D(256, 3, padding = 'same', activation='relu')
        
        #self.decoder3 = TransformerBlock(d_model * 4, num_heads, ff_dim, rate)
        self.upconv2 = layers.Conv3DTranspose(128, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same')
        self.x2_1 = Conv3D(128, 3, padding = 'same', activation='relu')
        self.x2_2 = Conv3D(128, 3, padding = 'same', activation='relu')
        
        #self.decoder2 = TransformerBlock(d_model * 2, num_heads, ff_dim, rate)
        self.upconv1 = layers.Conv3DTranspose(64, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same')
        self.x1_1 = Conv3D(64, 3, padding = 'same', activation='relu')
        self.x1_2 = Conv3D(64, 3, padding = 'same', activation='relu')
        
        #self.decoder1 = TransformerBlock(d_model, num_heads, ff_dim, rate)

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
        x = self.x4_1(x)
        x = self.x4_2(x)
        x = self.upconv3(x)
        x = self.x3_1(x)
        x = self.x3_2(x)
        x = self.upconv2(x)
        x = self.x2_1(x)
        x = self.x2_2(x)
        x = self.upconv1(x)
        x = self.x1_1(x)
        x = self.x1_2(x)
        x = self.output_layer(x)
        return x


def unet():
  model = TransformerUNet(image_size=96, patch_size=96)
  model.build(input_shape=(1, 96,96,96, 1))
      
  model.compile(optimizer = Adam(learning_rate = 1e-5, clipnorm=1), loss = 'MeanAbsoluteError', metrics = ['MeanSquaredError',PSNR, SSIM])
  
  model.summary(line_length=160)
  
  #if(pretrained_weights):
  #    model.load_weights(pretrained_weights)
  
  return model