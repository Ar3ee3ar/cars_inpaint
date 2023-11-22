import cv2
import numpy as np
from tensorflow import keras
from tqdm import tqdm

from tools.process_img import preprocess_img


## Ref: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.
class createAugment(keras.utils.Sequence):
  'Generates data for Keras'
  def __init__(self, X, y,mask_ds, batch_size=8, dim=(32, 32), n_channels=3, shuffle=True):
      'Initialization'
      self.batch_size = batch_size
      self.X = X/255.0
      self.y = y/255.0
      self.mask_ds = mask_ds/255.0
      self.dim = dim
      self.n_channels = n_channels
      self.shuffle = shuffle

      self.on_epoch_end()

  def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(len(self.X) / self.batch_size))

  def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

      # Generate data
      X_input_ds, y_output = self.__data_generation(indexes)
      return X_input_ds, y_output

  def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.X))
      if self.shuffle:
          np.random.shuffle(self.indexes)

  def __data_generation(self, idxs):
    # Masked_images is a matrix of masked images used as input
    Masked_images = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # Masked image
    # Mask_batch is a matrix of binary mask_ds used as input
    Mask_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # Binary Mask_dsmask_ds
    # y_batch is a matrix of original images used for computing error from reconstructed image
    y_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # Original image


    ## Iterate through random indexes
    for i, idx in enumerate(idxs):
      image_copy = self.X[idx].copy()

      ## Get mask associated to that image
      masked_image, mask = self.__createMask(image_copy,idx)

      Masked_images[i,] = masked_image
      Mask_batch[i,] = mask
      y_batch[i] = self.y[idx]

    ## Return mask as well because partial convolution require the same.
    return [Masked_images, Mask_batch], y_batch

  def __createMask(self, img, idx):
    ## Prepare masking matrix
    # mask = np.full((32,32,3), 255, np.uint8) ## White background
    # for _ in range(np.random.randint(1, 10)):
    #   # Get random x locations to start line
    #   x1, x2 = np.random.randint(1, 32), np.random.randint(1, 32)
    #   # Get random y locations to start line
    #   y1, y2 = np.random.randint(1, 32), np.random.randint(1, 32)
    #   # Get random thickness of the line drawn
    #   thickness = np.random.randint(1, 3)
    #   # Draw black line on the white mask
    #   cv2.line(mask,(x1,y1),(x2,y2),(0,0,0),thickness)

    # mask = convert_img
    # mask = cv2.bitwise_not(mask_img) # for only u-net
    mask = self.mask_ds[idx]
    ## Mask the image
    # masked_image = img.copy()
    masked_image = mask * img + (1 - mask) * 1.0
    # masked_image[mask==0] = 255 #bitwise
    # masked_image[mask==255] = 255 # original
    # masked_image[mask==255] = 255 # for only u-net

    return masked_image, mask

class Dataset:
    def __init__(self,config_list,input_dir,mask_dir,label_dir,image_size=(128,128)):
        self.config_list = config_list
        self.input_dir = input_dir
        self.mask_dir = mask_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.input_ds = []
        self.label_ds = []
        self.mask_ds = []
        self.config = []

    def read_config(self):
        with open(self.config_list) as f:
            for line in f.readlines():
                self.config.append(float(line.strip()))
    
    def process_data(self):
        self.read_config()
        with open(self.input_dir) as f:
            for count, line in enumerate(tqdm(f)):
                # print(line.strip())
                img = preprocess_img(line.strip(), self.config[count],self.image_size)
                self.input_ds.append(img)

        with open(self.mask_dir) as f:
            for count, line in enumerate(tqdm(f)):
                # print(line.strip())
                img = preprocess_img(line.strip(), self.config[count],self.image_size)
                img = cv2.bitwise_not(img)
                self.mask_ds.append(img)

        with open(self.label_dir) as f:
            for count, line in enumerate(tqdm(f)):
                # print(line.strip())
                img = preprocess_img(line.strip(), self.config[count],self.image_size)
                self.label_ds.append(img)
        
        self.input_ds = np.array(self.input_ds).astype('float32')
        self.label_ds = np.array(self.label_ds).astype('float32')
        self.mask_ds = np.array(self.mask_ds).astype('float32')
        return(self.input_ds,self.label_ds,self.mask_ds)