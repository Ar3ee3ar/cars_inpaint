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
from tools.utils import generate_images, wandb_log, view_test
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
    parser.add_argument("--weightG", "-model_weight_g",type=str, default='', help="path to generator model weight")
    parser.add_argument("--img_size", "-img_size",type=int, default=128, help="training image size")

    arg = parser.parse_args()
    return arg

def main():
    # print( _argparse().dir)
    # print(bool(strtobool(_argparse().wandb)))
    # print(_argparse().weightG)
    # print(_argparse().weightD)
    # print(_argparse().save)
    main_dir = _argparse().dir
    batch_size = 4

    # list of test dataset
    test_config_dir = main_dir + 'car_ds/data/test/config_zoom.txt'
    test_mask_dir = main_dir + 'car_ds/data/test/masks.txt'
    test_input_dir = main_dir + 'car_ds/data/test/masked_img.txt'
    test_label_dir = main_dir + 'car_ds/data/test/output.txt'

    # list of dataset
    x_test = []
    y_test = []
    mask_test = []

    image_size = (_argparse().img_size,_argparse().img_size)
    input_model_size = [_argparse().img_size,_argparse().img_size,3]

    test = Dataset(test_config_dir,test_input_dir,test_mask_dir,test_label_dir,image_size)
    x_test, y_test, mask_test = test.process_data()


    ## Prepare training and testing mask-image pair generator (with discriminator)
    testgen = createAugment(x_test, y_test,mask_test,batch_size=batch_size,dim=image_size, shuffle=False)

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

    view_test(testgen,generator,_argparse().model,batch_size)
       
    
if __name__ == '__main__':
    main()