import cv2
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

tf.config.run_functions_eagerly(True)

def preprocess_img (path_img, zoom, img_size=(128,128)):
  img = cv2.imread(path_img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  if(zoom != 0.0):
    img = np.array(tf.image.central_crop(img,zoom))
  img = cv2.resize(img, img_size)
  return img

def preprocess_vgg(image):
  process_img = tf.convert_to_tensor([tf.image.resize(x, [224,224], method='bilinear') for x in image], dtype=tf.float32)
  # process_img = tfio.experimental.color.rgb_to_bgr(process_img)
  # process_img = process_img * 255
  return process_img