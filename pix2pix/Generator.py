import tensorflow as tf

# from network import downsample, upsample
from .network import downsample, upsample

def Generator(input_shape = [256,256,3]):
  OUTPUT_CHANNELS = 3

  inputs = tf.keras.layers.Input(shape=input_shape)

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def dilated_Generator(input_shape = [512,512,3], neck=8):
  OUTPUT_CHANNELS = 3

  inputs = tf.keras.layers.Input(shape=input_shape)

  ##### Bottle neck 16x16 ------------------------------
  if(neck==16):
    down_stack = [
      downsample(64, 4, apply_batchnorm=False),  # (batch_size, 256, 256, 64)
      downsample(128, 4),  # (batch_size, 128, 128, 128)
      downsample(256, 4),  # (batch_size, 64, 64, 256)
      downsample(512, 4),  # (batch_size, 32, 32, 512)
      downsample(512, 4),  # (batch_size, 16, 16, 512)
      downsample(512, 3,stride=1,rate=(2,2)),  # (batch_size, 16, 16, 512) D
      downsample(512, 3,stride=1,rate=(2,2)),  # (batch_size, 16, 16, 512) D
      downsample(512, 3,stride=1,rate=(2,2)),  # (batch_size, 16, 16, 512) D
    ]

    up_stack = [
      upsample(512, 3,stride=1,rate=(2,2), apply_dropout=True),  # (batch_size, 16, 16, 512) D
      upsample(512, 3,stride=1,rate=(2,2), apply_dropout=True),  # (batch_size, 16, 16, 512) D
      upsample(512, 3,stride=1, apply_dropout=True),  # (batch_size, 16, 16, 1024)
      upsample(512, 4),  # (batch_size, 32, 32, 1024)
      upsample(256, 4),  # (batch_size, 64, 64, 512)
      upsample(128, 4),  # (batch_size, 128, 128, 256)
      upsample(64, 4),  # (batch_size, 256, 256, 128)
    ]
  ##### -----------------------------------

  ##### Bottle neck 8x8 -----------------------------
  elif(neck==8):
    down_stack = [
      downsample(64, 4, apply_batchnorm=False),  # (batch_size, 256, 256, 64)
      downsample(128, 4),  # (batch_size, 128, 128, 128)
      downsample(256, 4),  # (batch_size, 64, 64, 256)
      downsample(512, 4),  # (batch_size, 32, 32, 512)
      downsample(512, 4),  # (batch_size, 16, 16, 512)
      downsample(512, 3),  # (batch_size, 8, 8, 512) 
      downsample(512, 2,stride=1,rate=(2,2)),  # (batch_size, 8, 8, 512) D
      downsample(512, 2,stride=1,rate=(2,2)),  # (batch_size, 8, 8, 512) D
    ]

    up_stack = [
      upsample(512, 3,stride=1,rate=(2,2), apply_dropout=True),  # (batch_size, 8, 8, 512) D
      upsample(512, 3,stride=1, apply_dropout=True),  # (batch_size, 8, 8, 512)
      upsample(512, 4,apply_dropout=True),  # (batch_size, 16, 16, 1024)
      upsample(512, 4),  # (batch_size, 32, 32, 1024)
      upsample(256, 4),  # (batch_size, 64, 64, 512)
      upsample(128, 4),  # (batch_size, 128, 128, 256)
      upsample(64, 4),  # (batch_size, 256, 256, 128)
    ]
  ##### -----------------------------------

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


