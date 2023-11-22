import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import datetime
import os
import time
import wandb
import matplotlib.pyplot as plt
from distutils.util import strtobool

from dataset import Dataset, createAugment
from partial_conv.generator import dice_coef, InpaintingModel
from partial_conv.discriminator import Discriminator
from tools.utils import wandb_log, view_test
from tools.loss import discriminator_loss,generator_loss

# pix2pix
import pix2pix.Generator as p2pG
import pix2pix.Discriminator as p2pD

tf.config.run_functions_eagerly(True)

# Description: command ไว้ทำตามคำสั่ง
def _argparse():
    # print('parsing args...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-main_dir",type=str, default='', help="parent directory that store dataset")
    parser.add_argument("--model", "-model_name",type=str, default='', help="model name: pconv|p2p")
    parser.add_argument("--wandb", "-enable_wandb", type=str, nargs='?', const='True', default="False", help="enable wandb log")
    parser.add_argument("--weightG", "-model_weight_g",type=str, default='', help="path to generator model weight")
    parser.add_argument("--weightD", "-model_weight_d",type=str, default='', help="path to discriminator model weight")
    parser.add_argument("--save", "-save_step",type=int, default=500, help="saving model every save_step step")
    parser.add_argument("--step", "-train_step",type=int, default=10000, help="epoch")
    parser.add_argument("--lrG", "-lr_G",type=float, default=0.0002, help="learning rate of generator")
    parser.add_argument("--lrD", "-lr_D",type=float, default=0.0002, help="learning rate of discriminator")
    parser.add_argument("--img_size", "-img_size",type=int, default=128, help="training image size")
    parser.add_argument("--batch_size", "-batch_size",type=int, default=1, help="training image batch size")
    arg = parser.parse_args()
    return arg

class train_main:
  def __init__(self,traingen,generator,generator_optimizer,discriminator,discriminator_optimizer,
               example_masked_images,example_masks,example_sample_labels,
               checkpoint,checkpoint_prefix,summary_writer,
               steps, logs=None):
    # dataset
    self.traingen = traingen
    # model
    self.generator = generator
    self.discriminator = discriminator
    self.generator_optimizer = generator_optimizer
    self.discriminator_optimizer= discriminator_optimizer
    # example data
    self.example_masked_images = example_masked_images
    self.example_masks = example_masks
    self.example_sample_labels = example_sample_labels
    # tensorboard
    self.checkpoint = checkpoint
    self.checkpoint_prefix = checkpoint_prefix
    self.summary_writer = summary_writer
    # training step
    self.steps = steps
    # log path
    self.logs = logs

    self.fit()

  # training process
  @tf.function
  def train_step(self,input_image,mask,target,step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      if(_argparse().model == 'pconv'):
        gen_output = self.generator([input_image,mask], training=True)
        # replace img
        inpaint_img = mask * target + (1 - mask) * gen_output
        plt.imshow(inpaint_img[0])
        plt.savefig('test_save_train.png')
        inpaint_img = tf.convert_to_tensor(inpaint_img, dtype=tf.float32)

        # Discriminator
        disc_real_output = self.discriminator(target, training=True)
        disc_generated_output = self.discriminator(gen_output, training=True)
      elif(_argparse().model == 'p2p'):
        gen_output = self.generator(input_image, training=True)
        # normalize to [0,1]
        input_image = (input_image* 0.5) + 0.5
        target = (target* 0.5) + 0.5
        gen_output = (gen_output* 0.5) + 0.5
        # replace image
        inpaint_img = mask * target + (1 - mask) * gen_output
        inpaint_img = tf.convert_to_tensor(inpaint_img, dtype=tf.float32)
        # plot img
        display_list = [input_image[0], target[0], gen_output[0], inpaint_img[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image', 'Inpaint Image']

        for i in range(4):
          plt.subplot(1, 4, i+1)
          plt.title(title[i])
          # Getting the pixel values in the [0, 1] range to plot.
          plt.imshow(display_list[i])
          plt.axis('off')
        #plt.imshow(inpaint_img[0])
        plt.savefig('test_save_train.png')

        # Discriminator
        disc_real_output = self.discriminator([input_image, target], training=True)
        disc_generated_output = self.discriminator([input_image, gen_output], training=True)

      gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
      disc_loss, perc_loss, style_loss, tv_loss, hole_loss, valid_loss = discriminator_loss(disc_real_output, disc_generated_output, gen_output,inpaint_img,mask, target)


    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            self.generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                self.discriminator.trainable_variables)

    self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                            self.generator.trainable_variables))
    self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                self.discriminator.trainable_variables))

    with self.summary_writer.as_default():
      tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//10)
      tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//10)
      tf.summary.scalar('l1_hole_loss', hole_loss, step=step//10)
      tf.summary.scalar('l1_valid_loss', valid_loss, step=step//10)
      tf.summary.scalar('disc_loss', disc_loss, step=step//10)
      tf.summary.scalar('perc_loss', perc_loss, step=step//10)
      tf.summary.scalar('style_loss', style_loss, step=step//10)
      tf.summary.scalar('total_variation_loss', tv_loss, step=step//10)
      tf.summary.image("input img", input_image, step=step//10)
      tf.summary.image("predict img", gen_output, step=step//10)
      tf.summary.image("inpaint img", inpaint_img, step=step//10)


  def save_model(self,step):
    # "model.h5" is saved in wandb.run.dir & will be uploaded at the end of training
    name_g = step+'G_weight.h5'
    name_d = step+'D_weight.h5'
    self.generator.save(os.path.join(self.logs_path, name_g))
    self.discriminator.save(os.path.join(self.logs_path, name_d))
    # model.save('/content/drive/MyDrive/deepimageinpainting/logs/fit/20231008-075026/'+name)
    # Save a model file manually from the current directory:
    if(bool(strtobool(_argparse().wandb))):
      wandb.save(name_g)
      wandb.save(name_d)

  def generate_images(self):
    test_input = self.ex_masked_images
    mask = self.example_masks
    tar = self.example_sample_labels
    if(_argparse().model == "pconv"):
      gen_output = self.generator([test_input,mask], training=True)
      inpaint_img = [mask * tar[0] + (1 - mask) * gen_output[0]]
      display_list = [test_input[0], tar[0], gen_output[0],inpaint_img[0][0]]
    elif(_argparse().model == "p2p"):
      gen_output = self.generator(test_input, training=True)
      gen_output_norm = (gen_output[0] * 0.5) + 0.5
      test_input_norm = (test_input[0] * 0.5) + 0.5
      tar_norm = (tar[0] * 0.5) + 0.5
      inpaint_img = [mask * tar_norm + (1 - mask) * gen_output_norm]
      display_list = [test_input_norm, tar_norm, gen_output_norm,inpaint_img[0][0]]
    
    title = ['Input Image', 'Ground Truth', 'Predicted Image', 'Inpaint Image']

    for i in range(4):
      plt.subplot(1, 4, i+1)
      plt.title(title[i])
      # Getting the pixel values in the [0, 1] range to plot.
      plt.imshow(display_list[i])
      plt.axis('off')
    # plt.show()
    plt.savefig('test_save.png')
    if(bool(strtobool(_argparse().wandb))):
      wandb.log({"masked_images": [wandb.Image(display_list[0])]})
      wandb.log({"predictions": [wandb.Image(display_list[2])]})
      wandb.log({"inpaint": [wandb.Image(display_list[3])]})
      wandb.log({"labels": [wandb.Image(display_list[1])]})

  def fit(self):
    # for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    start = time.time()
    for step in range(self.steps):
      for i in range(len(self.traingen)):
        [masked_images, masks], sample_labels = self.traingen[i]
        input_image = masked_images.astype('float32')
        mask = masks.astype('float32')
        target = sample_labels.astype('float32')
        if(_argparse().model == 'p2p'):
          input_image = (input_image - 0.5)/0.5
          target = (target - 0.5)/0.5
        # for input_image, mask,target in zip(masked_images, masks, sample_labels):
          # input_image = tf.expand_dims(input_image, axis=0)
          # mask = tf.expand_dims(mask, axis=0)
          # target = tf.expand_dims(target, axis=0)
          # print(input_image.shape)
      if (step) % 10 == 0:
        # display.clear_output(wait=True)

        if step != 0:
          print(f'Time taken for 10 steps: {time.time()-start:.2f} sec\n')

        start = time.time()

        ex_masked_images = tf.expand_dims(self.example_masked_images[0], axis=0)
        ex_masks = tf.expand_dims(self.example_masks[0], axis=0)
        ex_sample_labels = tf.expand_dims(self.example_sample_labels[0], axis=0)
        if(_argparse().model == 'p2p'):
          ex_masked_images = (ex_masked_images - 0.5)/0.5
          ex_sample_labels = (ex_sample_labels - 0.5)/0.5
        # print(ex_masked_images)
        # print(ex_masks)
        # print(ex_sample_labels)
        self.generate_images()
        print(f"Step: {step//10}k")
        
        
      self.train_step(input_image,mask,target, step)

      # Training step
      if (step+1) % 10 == 0:
        print('.', end='', flush=True)


      # Save (checkpoint) the model every 5k steps
      if (step + 1) % _argparse().save == 0:
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        self.save_model(str(step))
        # self.save_model(self.generator,self.logs_path,str(step)+'G_weight.h5')
        # self.save_model(self.discriminator,self.logs_path,str(step)+'D_weight.h5')


def main():
    # print( _argparse().dir)
    # print(bool(strtobool(_argparse().wandb)))
    # print(_argparse().weightG)
    # print(_argparse().weightD)
    # print(_argparse().save)
    main_dir = _argparse().dir
    batch_size = _argparse().batch_size
    # print(bool(strtobool(_argparse().wandb)))
    if(bool(strtobool(_argparse().wandb))):
      log_path = wandb_log()
      # print(log_path)

    # main_dir = ''
    # list of training dataset
    train_config_dir = main_dir + 'car_ds/data/train/config_zoom.txt'
    train_mask_dir = main_dir + 'car_ds/data/train/masks.txt'
    train_input_dir = main_dir + 'car_ds/data/train/masked_img.txt'
    train_label_dir = main_dir + 'car_ds/data/train/output.txt'
    # list of test dataset
    test_config_dir = main_dir + 'car_ds/data/test/config_zoom.txt'
    test_mask_dir = main_dir + 'car_ds/data/test/masks.txt'
    test_input_dir = main_dir + 'car_ds/data/test/masked_img.txt'
    test_label_dir = main_dir + 'car_ds/data/test/output.txt'

    # list of dataset
    x_train = []
    y_train = []
    mask_train = []

    x_test = []
    y_test = []
    mask_test = []
    image_size = (_argparse().img_size,_argparse().img_size)
    input_model_size = [_argparse().img_size,_argparse().img_size,3]

    train = Dataset(train_config_dir,train_input_dir,train_mask_dir,train_label_dir,image_size)
    x_train, y_train, mask_train = train.process_data()
    # x_train = train.input_ds
    # y_train = train.label_ds
    # mask_train = train.mask_ds

    # test = Dataset(test_config_dir,test_input_dir,test_mask_dir,test_label_dir)
    # x_test, y_test, mask_test = test.process_data()
    # x_test = test.input_ds
    # y_test = test.label_ds
    # mask_test = test.mask_ds


    ## Prepare training and testing mask-image pair generator (with discriminator)
    traingen = createAugment(x_train, y_train,mask_train,batch_size=batch_size,dim=image_size)
    # print(type(traingen[0][0]))
    # testgen = createAugment(x_test, y_test,mask_test,batch_size=8,dim=(128,128), shuffle=False)

    # initial generator model
    keras.backend.clear_session()
    if(_argparse().model == 'pconv'):
      generator = InpaintingModel().prepare_model(input_size=input_model_size)
      generator.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[dice_coef])
      #keras.utils.plot_model(generator, show_shapes=True, dpi=60, to_file='model_v2_128.png')
      # initial discriminator model
      discriminator = Discriminator(input_size=input_model_size)
      #tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)
    elif(_argparse().model == 'p2p'):
      generator = p2pG.Generator(input_shape=input_model_size)
      # keras.utils.plot_model(generator, show_shapes=True, dpi=60, to_file='p2p_G_256.png')
      discriminator = p2pD.Discriminator(input_shape=input_model_size)
      # tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64, to_file='p2p_D_256.png')

    if(_argparse().weightG != ''):
      generator.load_weights(_argparse().weightG) # 500

    if(_argparse().weightD != ''):
      discriminator.load_weights(_argparse().weightD) # 500

    # define optimizer
    generator_optimizer = tf.keras.optimizers.Adam(_argparse().lrG, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(_argparse().lrD, beta_1=0.5)

    # define checkpoint
    log_dir= _argparse().dir + "logs/"
    time_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + time_now)

    checkpoint_dir = log_dir + "fit/" + time_now +'/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
    
    # if(bool(strtobool(_argparse().wandb))):
    #   logs_path = logs_path
    # else:
    #   logs_path = checkpoint_dir
    
    [example_masked_images, example_masks], example_sample_labels = traingen[54]
    example_masked_images = example_masked_images.astype('float32')
    example_masks = example_masks.astype('float32')
    example_sample_labels = example_sample_labels.astype('float32')
        
    if(bool(strtobool(_argparse().wandb))):
      train_main(traingen,generator,generator_optimizer,discriminator,discriminator_optimizer,
                   example_masked_images,example_masks,example_sample_labels,
                   checkpoint,checkpoint_prefix,summary_writer,logs= log_path,
                   steps=_argparse().step,
                   )
    else:
      train_main(traingen,generator,generator_optimizer,discriminator,discriminator_optimizer,
                   example_masked_images,example_masks,example_sample_labels,
                   checkpoint,checkpoint_prefix,summary_writer,logs= checkpoint_dir,
                   steps=_argparse().step)
    
if __name__ == '__main__':
    main()