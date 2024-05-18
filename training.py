import argparse
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import datetime
import os
import time
import wandb
import matplotlib.pyplot as plt
from distutils.util import strtobool
from tqdm import tqdm
from box import Box
import torch
from softadapt import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt

from dataset import Dataset, createAugment, prepare_other_car_mask
from partial_conv.generator import InpaintingModel
# from partial_conv.generator import dice_coef, InpaintingModel
# from partial_conv.discriminator import Discriminator
from tools.utils import wandb_log, view_test
from tools.loss import discriminator_loss,generator_loss, generator_l1_loss
from tools.metrics import metrics
# from tools.SoftAdapt.softadapt import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt
from config import cfg

# pix2pix
import pix2pix.Generator as p2pG
import pix2pix.Discriminator as p2pD
#import pix2pix.Discriminator_con as p2pD_con

tf.config.run_functions_eagerly(True)
# tf.experimental.numpy.experimental_enable_numpy_behavior()

# Description: for setting model
def _argparse():
    # print('parsing args...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-main_dir",type=str, default='', help="parent directory that store dataset")
    parser.add_argument("--config_file", "-config_file_name",type=str, default='', help="train test config data's file name")
    parser.add_argument("--model", "-model_name",type=str, default='', help="model name: pconv|p2p")
    parser.add_argument("--wandb", "-enable_wandb", type=str, nargs='?', const='True', default="False", help="enable wandb log")
    parser.add_argument("--ckpt_path", "-ckpt_path",type=str, default='', help="path to checkpoint folder")
    parser.add_argument("--weightG", "-model_weight_g",type=str, default='', help="path to Generator from phase 1 model weight")
    parser.add_argument("--save", "-save_step",type=int, default=500, help="saving model every save_step step")
    parser.add_argument("--step", "-train_step",type=int, default=10000, help="epoch")
    parser.add_argument("--lrG", "-lr_G",type=float, default=0.0002, help="learning rate of generator")
    parser.add_argument("--lrD", "-lr_D",type=float, default=0.0002, help="learning rate of discriminator")
    parser.add_argument("--img_size", "-img_size",type=int, default=128, help="training image size")
    parser.add_argument("--batch_size", "-batch_size",type=int, default=1, help="training image batch size")
    parser.add_argument("--bn", "-batch_norm", type=str, nargs='?', const='True', default="False", help="enable batch normalize")
    parser.add_argument("--random_mask", type=str, nargs='?', const='True', default="False", help="enable random mask")
    parser.add_argument("--random_car", type=str, nargs='?', const='True', default="False", help="enable random cars mask")
    parser.add_argument("--loss_type", "-loss_type",type=str, default='pconv', help="model name: pconv|p2p")
    parser.add_argument('--exclude_loss', '-loss-list', nargs='+', default=[], help = "perc | style | tv | valid | hole")
    parser.add_argument("--inpaint_mode", "-inpaint_mode",type=str, default='per', help="training model to inpaint per|cube view")
    parser.add_argument("--train_phase",type=int, default=1, help="training phase : 1(default)- non-adv | 2 - with adv")
    parser.add_argument("--cont", type=str, nargs='?', const='True', default="False", help="continue training in same phase")
    arg = parser.parse_args()
    return arg

class train_main:
  def __init__(self,traingen,valgen,generator,generator_optimizer,discriminator,discriminator_optimizer,
               example_masked_images,example_masks,example_sample_labels,
               checkpoint,checkpoint_prefix,summary_writer,
               steps, logs=None):
    # dataset
    self.traingen = traingen
    self.valgen = valgen
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
    # plot count (for loss graph)
    self.plot_count = 0
    # best loss for saving model
    self.best_gan_loss = 99999
    self.model_gan_loss = 0
    # phase training
    if(cfg.cont):
      self.phase = cfg.train_phase
    else:
      self.phase = 2
    # store lambda loss
    self.loss_lambda = {
        "LAMBDA_adv": 1,
        "LAMBDA_l1": 1,
        "LAMBDA_perc": 1,
        "LAMBDA_tv": 1,
        "LAMBDA_hole": 0,
        "LAMBDA_valid": 0,
        "LAMBDA_style": 1
    }

    self.fit()

  # training process
  @tf.function
  def train_step(self,input_image,mask,target,step):
    if(self.phase == 1):
      with tf.GradientTape() as gen_tape:
        # print('start phase 1')
        if(cfg.model == 'pconv'):
          gen_output = self.generator([input_image,mask], training=True)
          # replace img
          inpaint_img = mask * input_image + (1 - mask) * gen_output
          # plt.imshow(inpaint_img[0])
          # plt.savefig('test_save_train.png')
          inpaint_img = tf.convert_to_tensor(inpaint_img, dtype=tf.float32)

          # Discriminator
          # disc_real_output = self.discriminator(target, training=True)
          # disc_generated_output = self.discriminator(gen_output, training=True)
          # old
          # disc_real_output = self.discriminator([input_image, target], training=True)
          # disc_generated_output = self.discriminator([input_image, gen_output], training=True)
          # new
          # disc_real_output = self.discriminator( target, training=True)
          # disc_generated_output = self.discriminator( gen_output, training=True)
        elif(cfg.model == 'p2p'):
          gen_output = self.generator(input_image, training=True)
          # normalize to [0,1]
          input_image = (input_image* 0.5) + 0.5
          target = (target* 0.5) + 0.5
          gen_output = (gen_output* 0.5) + 0.5
          # replace image
          inpaint_img = mask * input_image + (1 - mask) * gen_output
          inpaint_img = tf.convert_to_tensor(inpaint_img, dtype=tf.float32)
          # plot img
        # display_list = [input_image[0], target[0], gen_output[0], inpaint_img[0]]
        # title = ['Input Image', 'Ground Truth', 'Predicted Image', 'Inpaint Image']

        # for i in range(4):
        #   plt.subplot(1, 4, i+1)
        #   plt.title(title[i])
        #   # Getting the pixel values in the [0, 1] range to plot.
        #   plt.imshow(display_list[i])
        #   plt.axis('off')
        # #plt.imshow(inpaint_img[0])
        # plt.savefig('test_save_train.png')
        gen_l1_loss = generator_l1_loss(gen_output, target,loss_lambda= self.loss_lambda)
        
        self.model_gan_loss = gen_l1_loss

      generator_gradients = gen_tape.gradient(gen_l1_loss,
                                              self.generator.trainable_variables)

      self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                              self.generator.trainable_variables))
      with self.summary_writer.as_default():
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step)
        # print('plot non-D')
      return [gen_l1_loss]
    elif(self.phase == 2):
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # print('start phase 2')
        if(cfg.model == 'pconv'):
          gen_output = self.generator([input_image,mask], training=True)
          # replace img
          inpaint_img = mask * input_image + (1 - mask) * gen_output
          # plt.imshow(inpaint_img[0])
          # plt.savefig('test_save_train.png')
          inpaint_img = tf.convert_to_tensor(inpaint_img, dtype=tf.float32)

          # Discriminator
          # disc_real_output = self.discriminator(target, training=True)
          # disc_generated_output = self.discriminator(gen_output, training=True)
          # old
          # disc_real_output = self.discriminator([input_image, target], training=True)
          # disc_generated_output = self.discriminator([input_image, gen_output], training=True)
          # new
          # disc_real_output = self.discriminator( target, training=True)
          # disc_generated_output = self.discriminator( gen_output, training=True)
        elif(cfg.model == 'p2p'):
          gen_output = self.generator(input_image, training=True)
          # normalize to [0,1]
          input_image = (input_image* 0.5) + 0.5
          target = (target* 0.5) + 0.5
          gen_output = (gen_output* 0.5) + 0.5
          # replace image
          inpaint_img = mask * input_image + (1 - mask) * gen_output
          inpaint_img = tf.convert_to_tensor(inpaint_img, dtype=tf.float32)
          # plot img
        # display_list = [input_image[0], target[0], gen_output[0], inpaint_img[0]]
        # title = ['Input Image', 'Ground Truth', 'Predicted Image', 'Inpaint Image']

        # for i in range(4):
        #   plt.subplot(1, 4, i+1)
        #   plt.title(title[i])
        #   # Getting the pixel values in the [0, 1] range to plot.
        #   plt.imshow(display_list[i])
        #   plt.axis('off')
        # #plt.imshow(inpaint_img[0])
        # plt.savefig('test_save_train.png')
        # Discriminator
        # old
        disc_real_output = self.discriminator([input_image, target], training=True)
        disc_generated_output = self.discriminator([input_image, gen_output], training=True)
        # new
        # disc_real_output = self.discriminator( target, training=True)
        # disc_generated_output = self.discriminator( gen_output, training=True)

        if(cfg.loss_type == 'pconv'):
          gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target,loss_lambda= self.loss_lambda)
          disc_total_loss, disc_loss, perc_loss, style_loss, tv_loss, hole_loss, valid_loss = discriminator_loss(disc_real_output, disc_generated_output, gen_output,inpaint_img,mask, target, loss_type = cfg.loss_type, loss_lambda= self.loss_lambda)
        elif(cfg.loss_type == 'p2p'):
          gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target,loss_lambda= self.loss_lambda)
          disc_total_loss = discriminator_loss(disc_real_output, disc_generated_output, gen_output,inpaint_img,mask, target, loss_type = cfg.loss_type,loss_lambda= self.loss_lambda)
        
        self.model_gan_loss = gen_l1_loss


      generator_gradients = gen_tape.gradient(gen_total_loss,
                                              self.generator.trainable_variables)
      discriminator_gradients = disc_tape.gradient(disc_total_loss,
                                                  self.discriminator.trainable_variables)

      self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                              self.generator.trainable_variables))
      self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                  self.discriminator.trainable_variables))

      with self.summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step)
        tf.summary.scalar('disc_total_loss', disc_total_loss, step=step)
        if(cfg.loss_type == 'pconv'):
          tf.summary.scalar('disc_loss', disc_loss, step=step)
          tf.summary.scalar('l1_hole_loss', hole_loss, step=step)
          tf.summary.scalar('l1_valid_loss', valid_loss, step=step)
          tf.summary.scalar('perc_loss', perc_loss, step=step)
          tf.summary.scalar('style_loss', style_loss, step=step)
          tf.summary.scalar('total_variation_loss', tv_loss, step=step)
        tf.summary.scalar('lambda_adv', self.loss_lambda["LAMBDA_adv"], step=step)
        tf.summary.scalar('lambda_l1', self.loss_lambda["LAMBDA_l1"], step=step)
        tf.summary.scalar('lambda_perc', self.loss_lambda["LAMBDA_perc"], step=step)
        tf.summary.scalar('lambda_tv', self.loss_lambda["LAMBDA_tv"], step=step)
        tf.summary.scalar('lambda_style', self.loss_lambda["LAMBDA_style"], step=step)
        # print('plot D')
        # tf.summary.image("input img", input_image, step=step//10)
        # tf.summary.image("predict img", gen_output, step=step//10)
        # tf.summary.image("inpaint img", inpaint_img, step=step//10)
      return [gen_gan_loss.numpy(), gen_l1_loss.numpy(), perc_loss.numpy(), tv_loss.numpy(), style_loss.numpy()]


  def save_model(self,step):
    # "model.h5" is saved in wandb.run.dir & will be uploaded at the end of training
    name_g = step+'G_weight.h5'
    name_d = step+'D_weight.h5'
    self.generator.save(os.path.join(self.logs, name_g))
    if(self.phase == 2 or cfg.model == 'p2p'):
      self.discriminator.save(os.path.join(self.logs, name_d))
    # model.save('/content/drive/MyDrive/deepimageinpainting/logs/fit/20231008-075026/'+name)
    # Save a model file manually from the current directory:
    if(cfg.wandb):
      wandb.save(name_g)
      if(self.phase == 2 or cfg.model == 'p2p'):
        wandb.save(name_d)

  def generate_images(self,test_input,mask,tar):
    # test_input = self.example_masked_images
    # mask = self.example_masks
    # tar = self.example_sample_labels
    if(cfg.model == "pconv"):
      gen_output = self.generator([test_input,mask], training=True)
      inpaint_img = [mask * tar[0] + (1 - mask) * gen_output[0]]
      display_list = [test_input[0], tar[0], gen_output[0],inpaint_img[0][0]]
    elif(cfg.model == "p2p"):
      gen_output = self.generator(test_input, training=True)
      gen_output_norm = (gen_output[0] * 0.5) + 0.5
      test_input_norm = (test_input[0] * 0.5) + 0.5
      tar_norm = (tar[0] * 0.5) + 0.5
      inpaint_img = [mask * tar_norm + (1 - mask) * gen_output_norm]
      display_list = [test_input_norm, tar_norm, gen_output_norm,inpaint_img[0][0]]
    
    # title = ['Input Image', 'Ground Truth', 'Predicted Image', 'Inpaint Image']

    # for i in range(4):
    #   plt.subplot(1, 4, i+1)
    #   plt.title(title[i])
    #   # Getting the pixel values in the [0, 1] range to plot.
    #   plt.imshow(display_list[i])
    #   plt.axis('off')
    # # plt.show()
    # plt.savefig('test_save.png')
    if(cfg.wandb):
      wandb.log({"predictions": [wandb.Image(display_list[2])],
                 "inpaint": [wandb.Image(display_list[3])]
                 })

  def val(self,step):
    l1_loss = 0
    l2_loss = 0
    psnr = 0
    ssim = 0
    for i in tqdm(range(2)):
      [masked_images, masks], sample_labels = self.valgen[i]
      input_image = masked_images.astype('float32')
      mask = masks.astype('float32')
      target = sample_labels.astype('float32')
      if(cfg.model == "pconv"):
        gen_output = self.generator([input_image,mask], training=True)
        # replace image
        inpaint_img = mask * input_image + (1 - mask) * gen_output
        inpaint_img = tf.convert_to_tensor(inpaint_img, dtype=tf.float32)
        metrics_value = metrics(gen_output.numpy(),target)
      elif(cfg.model == "p2p"):
        gen_output = self.generator(input_image, training=True)
        gen_output_norm = (gen_output * 0.5) + 0.5
        test_input_norm = (input_image * 0.5) + 0.5
        target_norm = (target *0.5) + 0.5
        # replace image
        inpaint_img = mask * test_input_norm + (1 - mask) * gen_output_norm
        inpaint_img = tf.convert_to_tensor(inpaint_img, dtype=tf.float32)
        metrics_value = metrics(gen_output_norm.numpy(),target_norm)
      l1_loss = l1_loss + metrics_value.mae()
      l2_loss = l2_loss + metrics_value.mse()
      psnr = psnr + metrics_value.psnr()
      ssim = ssim + metrics_value.ssim()
    
    with self.summary_writer.as_default():
      tf.summary.scalar('val_L1_loss', l1_loss/len(self.valgen), step=step)
      tf.summary.scalar('val_L2_loss', l2_loss/len(self.valgen), step=step)
      tf.summary.scalar('val_psnr', psnr/len(self.valgen), step=step)
      tf.summary.scalar('val_ssim', ssim/len(self.valgen), step=step)

  def fit(self):
    # for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    start = time.time()

    if(cfg.adapt_weight):
      # Change 1: Create a SoftAdapt object (with your desired variant)
      softadapt_object = LossWeightedSoftAdapt(beta=0.1)

      # Change 2: Define how often SoftAdapt calculate weights for the loss components
      epochs_to_make_updates = 2

      # Change 3: Initialize lists to keep track of loss values over the epochs we defined above
      loss_of_adv = []
      loss_of_l1 = []
      loss_of_perc = []
      loss_of_tv = []
      loss_of_style = []
      # Initializing adaptive weights to all ones.
      adapt_weights = torch.tensor([1,1,1,1,1])
    else:
      adapt_weights = torch.tensor([cfg.loss_lambda.LAMBDA_adv, cfg.loss_lambda.LAMBDA_l1, cfg.loss_lambda.LAMBDA_perc,
                                    cfg.loss_lambda.LAMBDA_tv, cfg.loss_lambda.LAMBDA_style])
    for step in tqdm(range(self.steps)):
      for i in tqdm(range(len(self.traingen))):
        [masked_images, masks], sample_labels = self.traingen[i]
        input_image = masked_images.astype('float32')
        mask = masks.astype('float32')
        target = sample_labels.astype('float32')
        if(cfg.model == 'p2p'):
          input_image = (input_image - 0.5)/0.5
          target = (target - 0.5)/0.5
      out_loss = self.train_step(input_image,mask,target, step)
      print(out_loss)
      # if step == 0 and i == len(self.traingen) - 1: # for 1 pic/epoch
      if step == 0 and i == 0: # for real training
          if(cfg.ckpt_path != ''):
            # print sum of initial weights for net
            print("Init Model Weights:", 
            sum([x.numpy().sum() for x in self.generator.weights]))
            print(tf.train.latest_checkpoint(cfg.ckpt_path))
            self.checkpoint.restore(tf.train.latest_checkpoint(cfg.ckpt_path)).assert_consumed()
            print("Checkpoint Weights:", 
            sum([x.numpy().sum() for x in self.checkpoint.generator.weights]))
            # print sum of weights for p2p & checkpoint after attempting to restore saved net 
            print("Restore Model Weights:", 
            sum([x.numpy().sum() for x in self.generator.weights]))
            print("Restored Checkpoint Weights:", 
            sum([x.numpy().sum() for x in self.checkpoint.generator.weights]))
            print("Done.")
            # generator.load_weights(cfg.weightG) # 500
          if(cfg.train_phase == 2 and not(cfg.cont)):
            self.phase = 2
        # self.plot_count = self.plot_count + 1
        # for input_image, mask,target in zip(masked_images, masks, sample_labels):
          # input_image = tf.expand_dims(input_image, axis=0)
          # mask = tf.expand_dims(mask, axis=0)
          # target = tf.expand_dims(target, axis=0)
          # print(input_image.shape)
      if(cfg.adapt_weight):
        loss_of_adv.append(out_loss[0])
        loss_of_l1.append(out_loss[1])
        loss_of_perc.append(out_loss[2])
        loss_of_tv.append(out_loss[3])
        loss_of_style.append(out_loss[4])
        # Change 4: Make sure `epochs_to_make_change` have passed before calling SoftAdapt.
        if step % epochs_to_make_updates == 0 and step != 0:
          adapt_weights = softadapt_object.get_component_weights(torch.tensor(loss_of_adv), 
                                                                  torch.tensor(loss_of_l1), 
                                                                  torch.tensor(loss_of_perc),
                                                                  torch.tensor(loss_of_tv),
                                                                  torch.tensor(loss_of_style),
                                                                  verbose=False,
                                                                  )
          # Resetting the lists to start fresh (this part is optional)
          loss_of_adv = []
          loss_of_l1 = []
          loss_of_perc = []
          loss_of_tv = []
          loss_of_style = []

      self.loss_lambda["LAMBDA_adv"] = adapt_weights[0]
      self.loss_lambda["LAMBDA_l1"] = adapt_weights[1]
      self.loss_lambda["LAMBDA_perc"] = adapt_weights[2]
      self.loss_lambda["LAMBDA_tv"] = adapt_weights[3]
      self.loss_lambda["LAMBDA_style"] = adapt_weights[4]
        
      if (step) % 1 == 0:
        # display.clear_output(wait=True)

        if step != 0:
          print(f'Time taken for 10 steps: {time.time()-start:.2f} sec | weight: {self.loss_lambda["LAMBDA_adv"]}, {self.loss_lambda["LAMBDA_l1"]}, {self.loss_lambda["LAMBDA_perc"]}, {self.loss_lambda["LAMBDA_tv"]}, {self.loss_lambda["LAMBDA_style"]} ')

        start = time.time()

        ex_masked_images = tf.expand_dims(self.example_masked_images[0], axis=0)
        ex_masks = tf.expand_dims(self.example_masks[0], axis=0)
        ex_sample_labels = tf.expand_dims(self.example_sample_labels[0], axis=0)
        if(cfg.model == 'p2p'):
          ex_masked_images = (ex_masked_images - 0.5)/0.5
          ex_sample_labels = (ex_sample_labels - 0.5)/0.5
        # print(ex_masked_images)
        # print(ex_masks)
        # print(ex_sample_labels)
        self.generate_images(ex_masked_images,ex_masks,ex_sample_labels)
        self.val(step)
        print(f"Step: {step//1}")

      # Training step
      if (step+1) % 1 == 0:
        print('.', end='', flush=True)

      # save best model
      if self.model_gan_loss < self.best_gan_loss:
        self.best_gan_loss = self.model_gan_loss
        self.save_model('best')
      # Save (checkpoint) the model every 5k steps
      if (step + 1) % cfg.save == 0:
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        self.save_model(str(step))
        # self.save_model(self.generator,self.logs_path,str(step)+'G_weight.h5')
        # self.save_model(self.discriminator,self.logs_path,str(step)+'D_weight.h5')


def main(cfg):
    print('phase: ',cfg.train_phase,' continue status: ',cfg.cont)
    # print(bool(strtobool(cfg.wandb)))
    # print(cfg.weightG)
    # print(cfg.weightD)
    # print(cfg.save)
    main_dir = cfg.dir
    config_folder = cfg.config_file
    batch_size = cfg.batch_size
    # print(bool(strtobool(cfg.wandb)))
    if(cfg.wandb):
      log_path = wandb_log(cfg)
      # wandb.save(os.path.join(log_path, "config.py"))

      # print(log_path)

    # main_dir = ''
    # list of training dataset
    train_mask_dir = main_dir + 'car_ds/train_test_config/'+config_folder+'/val/masks.txt'
    train_input_dir = main_dir + 'car_ds/train_test_config/'+config_folder+'/val/input.txt'
    train_label_dir = main_dir + 'car_ds/train_test_config/'+config_folder+'/val/output.txt'
    # list of test dataset
    # train_config_dir = main_dir + 'car_ds/train_test_config/'+config_folder+'/test/config_zoom.txt'
    # train_mask_dir = main_dir + 'car_ds/train_test_config/'+config_folder+'/test/masks.txt'
    # train_input_dir = main_dir + 'car_ds/train_test_config/'+config_folder+'/test/masked_img.txt'
    # train_label_dir = main_dir + 'car_ds/train_test_config/'+config_folder+'/test/output.txt'
    # list of validate dataset
    val_mask_dir = main_dir + 'car_ds/train_test_config/'+config_folder+'/val/masks.txt'
    val_input_dir = main_dir + 'car_ds/train_test_config/'+config_folder+'/val/input.txt'
    val_label_dir = main_dir + 'car_ds/train_test_config/'+config_folder+'/val/output.txt'

    # list of dataset
    x_train = []
    y_train = []
    mask_train = []

    x_test = []
    y_test = []
    mask_test = []
    image_size = (cfg.img_size,cfg.img_size)
    input_model_size = [cfg.img_size,cfg.img_size,3]

    if(cfg.inpaint_mode == 'per'):
      train = Dataset(main_dir = main_dir,input_dir = train_input_dir,mask_dir = train_mask_dir,label_dir = train_label_dir,image_size = image_size)
      val = Dataset(main_dir = main_dir,input_dir = val_input_dir,mask_dir = val_mask_dir,label_dir = val_label_dir,image_size = image_size)
    if(cfg.inpaint_mode == 'cube'):
      train = Dataset(main_dir = main_dir,input_dir = train_input_dir,mask_dir=None,label_dir = train_label_dir,image_size = image_size)
      val = Dataset(main_dir = main_dir,input_dir = val_input_dir,mask_dir=None,label_dir = val_label_dir,image_size = image_size)
    x_train, y_train, mask_train = train.process_data()
    x_val, y_val, mask_val = val.process_data()
    x_train = train.input_ds
    y_train = train.label_ds
    mask_train = train.mask_ds

    # test = Dataset(test_config_dir,test_input_dir,test_mask_dir,test_label_dir)
    # x_test, y_test, mask_test = test.process_data()
    # x_test = test.input_ds
    # y_test = test.label_ds
    # mask_test = test.mask_ds

    if(cfg.random_car):
      other_car_mask = prepare_other_car_mask(main_dir,image_size)
    else:
      other_car_mask = None

    ## Prepare training and testing mask-image pair generator (with discriminator)
    traingen = createAugment(x_train, y_train, mask_train, batch_size=batch_size, dim=image_size, random_mask=cfg.random_mask, other_car_list=other_car_mask)
    valgen = createAugment(x_val, y_val, mask_val, batch_size=batch_size, dim=image_size, random_mask=cfg.random_mask, other_car_list=other_car_mask)

    # testgen = createAugment(x_test, y_test,mask_test,batch_size=8,dim=(128,128), shuffle=False)

    # initial generator model
    keras.backend.clear_session()
    if(cfg.model == 'pconv'):
      # old pconv -----------------------
      # generator = InpaintingModel().prepare_model(input_size=input_model_size)
      # generator.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[dice_coef])
      # #keras.utils.plot_model(generator, show_shapes=True, dpi=60, to_file='model_v2_128.png')
      # # initial discriminator model
      # discriminator = Discriminator(input_size=input_model_size)
      # #tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)
      # new pconv -------------------------
      generator = InpaintingModel().build_pconv_unet(input_size=input_model_size,train_bn = cfg.wandb)
      # keras.utils.plot_model(generator, show_shapes=True, dpi=60, to_file='new_pconv_128.png')
      discriminator = p2pD.Discriminator_con(input_shape=input_model_size)
    elif(cfg.model == 'p2p'):
      generator = p2pG.Generator(input_shape=input_model_size)
      # keras.utils.plot_model(generator, show_shapes=True, dpi=60, to_file='p2p_G_256.png')
      discriminator = p2pD.Discriminator_con(input_shape=input_model_size)
      # tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64, to_file='p2p_D_256.png')

    # define optimizer
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate = tf.Variable(cfg.lrG), beta_1=tf.Variable(0.5))
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = tf.Variable(cfg.lrD), beta_1=tf.Variable(0.5))

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

    # define checkpoint
    log_dir= cfg.dir + "logs/"
    time_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + time_now)

    checkpoint_dir = log_dir + "fit/" + time_now +'/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    
    # if(bool(strtobool(cfg.wandb))):
    #   logs_path = logs_path
    # else:
    #   logs_path = checkpoint_dir
    
    [example_masked_images, example_masks], example_sample_labels = traingen[54]
    example_masked_images = example_masked_images.astype('float32')
    example_masks = example_masks.astype('float32')
    example_sample_labels = example_sample_labels.astype('float32')

    if(cfg.wandb):
      wandb.log({"masked_images": [wandb.Image(example_masked_images[0])],
                 "labels": [wandb.Image(example_sample_labels[0])]
                 })
      train_main(traingen,valgen,generator,generator_optimizer,discriminator,discriminator_optimizer,
                   example_masked_images,example_masks,example_sample_labels,
                   checkpoint,checkpoint_prefix,summary_writer,logs= log_path,
                   steps=cfg.step,
                   )
    else:
      train_main(traingen,valgen,generator,generator_optimizer,discriminator,discriminator_optimizer,
                   example_masked_images,example_masks,example_sample_labels,
                   checkpoint,checkpoint_prefix,summary_writer,logs= checkpoint_dir,
                   steps=cfg.step)
    
if __name__ == '__main__':
    main(cfg)