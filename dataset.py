import cv2
import numpy as np
from tensorflow import keras
from tqdm import tqdm
import tensorflow as tf

from tools.process_img import preprocess_img, move_img, flip_img, zoom_img, crop_center


## Ref: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.
class createAugment(keras.utils.Sequence):
  'Generates data for Keras'
  def __init__(self, X, y,mask_ds, batch_size=8, dim=(32, 32), n_channels=3, 
               shuffle=True, random_mask = False, random_box = False, random_rotate = False,
               crop_size = 0, crop_method = "", other_car_list = None, training= True):
      'Initialization'
      self.batch_size = batch_size
      self.X = X/255.0
      self.y = y/255.0
      if mask_ds is not None:
        # mask_ds[mask_ds>0] = 255.0
        # mask_ds[mask_ds<=0] = 0.0
        self.mask_ds = mask_ds/255.0
      else:
        self.mask_ds = mask_ds
      self.dim = dim
      self.n_channels = n_channels
      self.shuffle = shuffle
      self.crop_size = crop_size
      self.crop_method = crop_method
      self.random_mask = random_mask
      self.random_box = random_box
      self.random_rotate = random_rotate
      self.training = training
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

  def random_crop(self,input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, self.crop_size, self.crop_size, 3])

    return cropped_image[0], cropped_image[1]

  def __data_generation(self, idxs):
    if(self.crop_size <= 0):
        # Masked_images is a matrix of masked images used as input
        Masked_images = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # Masked image
        # Mask_batch is a matrix of binary mask_ds used as input
        Mask_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # Binary Mask_dsmask_ds
        # y_batch is a matrix of original images used for computing error from reconstructed image
        y_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # Original image
    else:
       # Masked_images is a matrix of masked images used as input
       Masked_images = np.empty((self.batch_size, self.crop_size, self.crop_size, self.n_channels)) # Masked image
       # Mask_batch is a matrix of binary mask_ds used as input
       Mask_batch = np.empty((self.batch_size, self.crop_size, self.crop_size, self.n_channels)) # Binary Mask_dsmask_ds
       # y_batch is a matrix of original images used for computing error from reconstructed image
       y_batch = np.empty((self.batch_size, self.crop_size, self.crop_size, self.n_channels)) # Original image


    ## Iterate through random indexes
    for i, idx in enumerate(idxs):
      if(self.training):
        if(self.training or self.random_rotate):
            rand_rotate = np.random.randint(1, 4)
            image_copy = tf.image.rot90(self.X[idx].copy(), k=rand_rotate)
            real_img_copy = tf.image.rot90(self.y[idx].copy(), k=rand_rotate)
            input_img = image_copy
            real_img = real_img_copy
        else:
            input_img =  self.X[idx].copy()
            real_img = self.y[idx].copy()
        
        if(self.crop_size > 0):
            if(self.crop_method == "random"):
                input_img, real_img = self.random_crop(input_img, real_img)
            elif(self.crop_method == "center"):
                input_img = crop_center(input_img,(self.crop_size,self.crop_size))
                real_img = crop_center(real_img,(self.crop_size,self.crop_size))
            else:
                input_img = input_img
                real_img = real_img

        if(bool(np.random.randint(0, 2))):
                # Random mirroring
                input_img = tf.image.flip_left_right(input_img)
                real_img = tf.image.flip_left_right(real_img)
      else:
        input_img =  self.X[idx].copy()
        real_img = self.y[idx].copy()

      ## Get mask associated to that image
      masked_image, mask_img = self.__createMask(input_img,idx)

      Masked_images[i,] = masked_image
      Mask_batch[i,] = mask_img
      y_batch[i] = real_img

    ## Return mask as well because partial convolution require the same.
    return [Masked_images, Mask_batch], y_batch
  

  def __createMask(self, img, idx):
    ## Prepare masking matrix
    if(self.mask_ds is not None):
        if(self.crop_size > 0):
            mask = cv2.resize(self.mask_ds[idx], (self.dim[0],self.dim[1]))
            mask[mask != 1] = 0
        else:
            mask = self.mask_ds[idx]
    elif(self.mask_ds is None):
        mask = np.full((self.dim[0],self.dim[1],3), 1.0, np.float32) ## White background
    if(self.random_car):
        for _ in range(np.random.randint(1, 3)):
            other_car = self.other_car_list[np.random.randint(0, len(self.other_car_list))]
            if(self.crop_size > 0):
                other_car = cv2.resize(other_car, (self.crop_size,self.crop_size))
            else:
                other_car = other_car
            if(bool(np.random.randint(0, 2))): # move_img
                other_car = move_img(other_car,np.random.randint(-50, 50),np.random.randint(0, 50))
            if(bool(np.random.randint(0, 2))): # zoom
                other_car = zoom_img(other_car, zoom_factor=np.random.uniform(0.5,2.0))
            if(bool(np.random.randint(0, 2))): # flip
                other_car = flip_img(other_car)
            mask = mask + other_car
            mask = (mask > 1).astype(np.float32)
            # mask = mask - 1.0
    if(self.random_box):
        for _ in range(np.random.randint(1,3)):
            # random width and height of rectangle
            rect_width = np.random.randint(20,70)  # real_mask = (w,h) = (119,291)
            rect_height = np.random.randint(20,120)
            # set center point
            cx = np.random.randint(1, self.dim[0])
            cy = np.random.randint(1, self.dim[1])
            # Get random x locations to start line
            x1,x2 = cx -rect_width, cx +rect_width
            # Get random y locations to start line
            y1, y2 = cy -rect_height, cy +rect_height

            cv2.rectangle(mask, (x1,y1), (x2,y2), (0,0,0), -1)

    if(self.random_mask):
        # Set size scale
        size = int((self.dim[0] + self.dim[1]) * 0.03)
        # start_rand_y_half = self.dim[1]/2
        # start_rand_y_full = 1
        if self.dim[0] < 64 or self.dim[1] < 64:
            raise Exception("Width and Height of mask must be at least 64!")
        # Draw random lines
        for _ in range(np.random.randint(1, 10)):
            # Get random x locations to start line
            x1, x2 = np.random.randint(1, self.dim[0]), np.random.randint(1, self.dim[0])
            # Get random y locations to start line
            y1, y2 = np.random.randint(1, self.dim[1]), np.random.randint(1, self.dim[1])
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
    # mask[mask != 0] = 1
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
                    # img[img>0] = 255
                    # img[img<=0] = 0
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
    
def prepare_other_car_mask(main_dir,image_size,other_car_list=['random_car.txt']):
    other_car_pic = []
    for car_name in other_car_list:
        with open(main_dir + 'car_ds/train_test_config/'+car_name) as f:
            for count, line in enumerate(f):
                img = preprocess_img(line.strip(),image_size)
                img = cv2.bitwise_not(img)
                other_car_pic.append(img)
    other_car_pic = np.array(other_car_pic).astype('float32')
    print(len(other_car_pic))

    return other_car_pic