# from tensorflow.keras.layers import Conv2D
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Concatenate, Dense, LeakyReLU, Flatten

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def Discriminator(input_size=(128,128,3)):
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(input_size, name='image')
  # tar = tf.keras.layers.Input(shape=[32, 32, 3], name='target_image')

  # x = tf.keras.layers.concatenate(inp)  # (batch_size, 32, 32, channels*2)

  down1 = downsample(64, 4, False)(inp)  # (batch_size, 16, 16, 64)
  batchnorm1 = tf.keras.layers.BatchNormalization()(down1)
  leaky_relu1 = tf.keras.layers.LeakyReLU()(batchnorm1)

  down2 = downsample(128, 4)(leaky_relu1)  # (batch_size, 8, 8, 128)
  batchnorm2 = tf.keras.layers.BatchNormalization()(down2)
  leaky_relu2 = tf.keras.layers.LeakyReLU()(batchnorm2)

  down3 = downsample(256, 4)(leaky_relu2)  # (batch_size, 4, 4, 256)
  batchnorm3 = tf.keras.layers.BatchNormalization()(down3)
  leaky_relu3 = tf.keras.layers.LeakyReLU()(batchnorm3)

  down4 = downsample(512, 4)(leaky_relu3) # (batch_size,2,2,512)
  batchnorm4 = tf.keras.layers.BatchNormalization()(down4)
  leaky_relu4 = tf.keras.layers.LeakyReLU()(batchnorm4)

  # zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  # conv = tf.keras.layers.Conv2D(512, 4, strides=1,
  #                               kernel_initializer=initializer,
  #                               use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  # zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  down5 = tf.keras.layers.Conv2D(512, 2, strides=1,
                                 activation='sigmoid',
                                kernel_initializer=initializer)(leaky_relu4)  # (batch_size, 30, 30, 1)
  flatten = Flatten()(down5)
  output = Dense(1)(flatten)
  # model = Model(inputs=input_x_layer, outputs=output)

  return tf.keras.Model(inputs=inp, outputs=output)