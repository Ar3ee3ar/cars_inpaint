import cv2
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

tf.config.run_functions_eagerly(True)

def preprocess_img (path_img, zoom, img_size=(128,128)):
  # print(type(path_img))
  if isinstance(path_img, str): # input 
    img = cv2.imread(path_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # elif isinstance(path_img, (np.ndarray, np.generic) ):
  else:
    img = path_img
  if(zoom != 0.0):
    img = np.array(tf.image.central_crop(img,zoom))
  img = cv2.resize(img, img_size)
  return img

def preprocess_vgg(image):
  process_img = tf.convert_to_tensor([tf.image.resize(x, [224,224], method='bilinear') for x in image], dtype=tf.float32)
  # process_img = tfio.experimental.color.rgb_to_bgr(process_img)
  # process_img = process_img * 255
  return process_img

def crop_center(image,new_img_size=(190,290)): # new_width=190,new_height=290
  center = image.shape
  new_width = new_img_size[0]
  new_height = new_img_size[1]
  x = center[1]/2 - new_width/2
  y = center[0]/2 - new_height/2

  crop_img = image[int(y):int(y+new_height), int(x):int(x+new_width)]
  return crop_img

def resize(img,new_width,new_height):
   resize_img = cv2.resize(img, [new_width,new_height], interpolation = cv2.INTER_AREA)
   return resize_img

def pad_images_to_same_size(images,new_image_width = 1000,new_image_height=1000):
    old_image_height, old_image_width, channels = images.shape

    # create new image of desired size and color (blue) for padding
    color = (0,0,0)
    result = np.full((new_image_height,new_image_width, channels), color, dtype=np.float32)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = images
    # plt.imshow(result)
    # plt.show()
    return result