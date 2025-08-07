import tensorflow as tf
from tensorflow.keras.layers import Lambda, Conv2D, BatchNormalization, Activation, Input, MaxPool2D, ZeroPadding2D, SpatialDropout2D, MaxPool2D, UpSampling2D
from tensorflow.keras import Model
from tensorflow.keras.applications import DenseNet121
import sys
sys.path.append('/root/.local/lib/python3.10/site-packages')
import os
os.environ['SM_FRAMEWORK']='tf.keras'
import segmentation_models as sm

# Questo blocco fa da ponte con il base model. I parametri vanno settati di
# conseguenza.
def FirstConvolution(X, input_feature):
  num_init_features = input_feature
  X = ZeroPadding2D(padding=2)(X)
  X = Conv2D(filters = num_init_features, kernel_size=7, strides=2, use_bias=False)(X)
  X = Activation('relu')(X)
  X = MaxPool2D(pool_size=3, strides=2, padding='same')(X)
  return X

def DenseAsppBlock(X, num1, num2, dilation_rate, dropout,
                   bn_first=True):
  if bn_first:
    X = BatchNormalization(axis=-1)(X)
  X = Activation('relu')(X)
  X = Conv2D(filters=num1, kernel_size=1)(X)
  X = BatchNormalization(axis=-1)(X)
  X = Activation('relu')(X)
  X = ZeroPadding2D(padding=dilation_rate)(X)
  X = Conv2D(filters=num2, kernel_size=3, dilation_rate=dilation_rate)(X)
  if dropout > 0:
    X = SpatialDropout2D(rate=dropout)(X)
  return X

def Classification(X, dropout, num_class):
  X = SpatialDropout2D(rate=dropout)(X)
  X = Conv2D(filters=num_class, kernel_size=1, padding='valid', activation='sigmoid')(X)
  X = UpSampling2D(size=(32,32), interpolation='bilinear')(X)
  return X

def DenseAspp(num_classes, short=False, input_feature=512):

  X_input = Input((None, None, input_feature))
  X = X_input
  X = FirstConvolution(X, input_feature)

  # ASPP blocks
  aspp3 = DenseAsppBlock(X, input_feature/2, input_feature/8, 3, 0, False)
  X = tf.concat([aspp3, X], axis=-1)

  aspp6 = DenseAsppBlock(X, input_feature/2, input_feature/8, 6, 0, True)
  X = tf.concat([aspp6, X], axis=-1)

  aspp12 = DenseAsppBlock(X, input_feature/2, input_feature/8, 12, 0, True)
  X = tf.concat([aspp12, X], axis=-1)

  aspp18 = DenseAsppBlock(X, input_feature/2, input_feature/8, 18, 0, True)
  X = tf.concat([aspp18, X], axis=-1)

  if  not(short):
    aspp24 = DenseAsppBlock(X, input_feature/2, input_feature/8, 24, 0, True)
    X = tf.concat([aspp24, X], axis=-1)

  #classification
  X = Classification(X, 0.2, num_classes)

  model = Model(inputs=X_input, outputs=X, name='DenseAspp')
  return model

# Set encoder freeze to true to train only the DenseASPP module
def FullModel(num_classes, short=False, encoder_freeze=False):

  #The backbone is downloaded from Segmentation Model library
  model = sm.PSPNet('densenet121', downsample_factor=8,
                    encoder_freeze=encoder_freeze)
  intermediate_layer_name = 'pool3_relu'
  intermediate_layer = model.get_layer(intermediate_layer_name).output
  densenet121 = Model(inputs=model.input, outputs=intermediate_layer,
                      name='Densenet121')

  X_input = densenet121.input
  X = densenet121(X_input)
  dense_aspp = DenseAspp(num_classes, short=short)
  X = dense_aspp(X)
  model = Model(inputs=X_input, outputs=X, name='DenseASPP_with_backbone')

  return model
