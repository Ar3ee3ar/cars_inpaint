import cv2
import numpy as np
from tensorflow import keras
from tqdm import tqdm

from tools.process_img import preprocess_img, move_img, flip_img, zoom_img


## Ref: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.
class createAugment(keras.utils.Sequence):
  'Generates data for Keras'
  def __init__(self, X, y,mask_ds, batch_size=8, dim=(32, 32), n_channels=3, shuffle=True, random_mask = False, other_car_list = None):
      'Initialization'
      self.batch_size = batch_size
      self.X = X/255.0
      self.y = y/255.0
      if mask_ds is not None:
        self.mask_ds = mask_ds/255.0
      else:
        self.mask_ds = mask_ds
      self.dim = dim
      self.n_channels = n_channels
      self.shuffle = shuffle
      self.random_mask = random_mask
      if other_car_list is not None:
        self.random_car = True
        self.other_car_list = other_car_list/255.0
      else:
        self.random_car = False

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
    if(self.mask_ds is not None):
        mask = self.mask_ds[idx]
    elif(self.mask_ds is None):
        mask = np.full((self.dim[0],self.dim[1],3), 1.0, np.float32) ## White background
    if(self.random_car):
        for _ in range(np.random.randint(1, 3)):
            other_car = self.other_car_list[np.random.randint(0, len(self.other_car_list)-1)]
            if(bool(np.random.randint(0, 1))): # move_img
                other_car = move_img(other_car,np.random.randint(-50, 50),np.random.randint(-50, 50))
            if(bool(np.random.randint(0, 1))): # zoom
                other_car = zoom_img(other_car, zoom_factor=np.random.uniform(0.5,2.0))
            if(bool(np.random.randint(0, 1))): # flip
                other_car = flip_img(other_car)
            mask = mask + other_car
            mask = (mask > 1).astype(np.float32)
            # mask = mask - 1.0
    # Set size scale
    if(self.random_mask):
        size = int((self.dim[0] + self.dim[1]) * 0.03)
        if self.dim[0] < 64 or self.dim[1] < 64:
            raise Exception("Width and Height of mask must be at least 64!")
        # Draw random lines
        for _ in range(np.random.randint(1, 10)):
            # Get random x locations to start line
            x1, x2 = np.random.randint(1, self.dim[0]), np.random.randint(1, self.dim[1])
            # Get random y locations to start line
            y1, y2 = np.random.randint(1, self.dim[0]), np.random.randint(1, self.dim[1])
            # Get random thickness of the line drawn
            thickness = np.random.randint(1, 3)
            # Draw black line on the white mask
            cv2.line(mask,(x1,y1),(x2,y2),(0,0,0),thickness)
            # Draw random circles
        for _ in range(np.random.randint(1, 20)):
            x1, y1 = np.random.randint(1, self.dim[0]), np.random.randint(1, self.dim[1])
            radius = np.random.randint(3, size)
            cv2.circle(mask,(x1,y1),radius,(0,0,0), -1)
        # Draw random ellipses
        for _ in range(np.random.randint(1, 20)):
            x1, y1 = np.random.randint(1, self.dim[0]), np.random.randint(1, self.dim[1])
            s1, s2 = np.random.randint(1, self.dim[0]), np.random.randint(1, self.dim[1])
            a1, a2, a3 = np.random.randint(3, 180), np.random.randint(3, 180), np.random.randint(3, 180)
            thickness = np.random.randint(3, size)
            cv2.ellipse(mask, (x1,y1), (s1,s2), a1, a2, a3,(0,0,0), thickness)


    # mask = convert_img
    # mask = cv2.bitwise_not(mask_img) # for only u-net
    # mask = self.mask_ds[idx]
    ## Mask the image
    # masked_image = img.copy()
    masked_image = mask * img + (1 - mask) * 1.0
    # masked_image[mask==0] = 255 #bitwise
    # masked_image[mask==255] = 255 # original
    # masked_image[mask==255] = 255 # for only u-net

    return masked_image, mask

class Dataset:
    def __init__(self,main_dir,input_dir,mask_dir,label_dir,image_size=(128,128)):
        self.main_dir = main_dir
        self.input_dir = input_dir
        self.mask_dir = mask_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.input_ds = []
        self.label_ds = []
        self.mask_ds = []

    
    def process_data(self):
        with open(self.input_dir) as f:
            for count, line in enumerate(tqdm(f)):
                # print(line.strip())
                img = preprocess_img(self.main_dir+line.strip() ,self.image_size)
                self.input_ds.append(img)
        self.input_ds = np.array(self.input_ds).astype('float32')

        if self.mask_dir is not None:
            with open(self.mask_dir) as f:
                for count, line in enumerate(tqdm(f)):
                    # print(line.strip())
                    img = preprocess_img(self.main_dir+ line.strip(),self.image_size)
                    img = cv2.bitwise_not(img)
                    self.mask_ds.append(img)
            self.mask_ds = np.array(self.mask_ds).astype('float32')
        else:
            self.mask_ds = None

        with open(self.label_dir) as f:
            for count, line in enumerate(tqdm(f)):
                # print(line.strip())
                img = preprocess_img(self.main_dir + line.strip() ,self.image_size)
                self.label_ds.append(img)
        self.label_ds = np.array(self.label_ds).astype('float32')

        return(self.input_ds,self.label_ds,self.mask_ds)
    
def prepare_other_car_mask(main_dir,image_size):
    other_car_list = []
    with open(main_dir + 'car_ds/train_test_config/random_car.txt') as f:
      for count, line in enumerate(f):
          img = preprocess_img(line.strip(),image_size)
          img = cv2.bitwise_not(img)
          other_car_list.append(img)
    other_car_list = np.array(other_car_list).astype('float32')

    return other_car_list