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
from tools.process_img import preprocess_img
from tools.process_pano import Equirectangular,Perspective,inpaint_pano

tf.config.run_functions_eagerly(True)

# Description: command ไว้ทำตามคำสั่ง
def _argparse():
    # print('parsing args...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-main_dir",type=str, default='', help="parent directory that store dataset")
    parser.add_argument("--img", "-img_dir",type=str, default='', help="directory that store input") 
    parser.add_argument("--mask", "-mask_dir",type=str, default='', help="directory that store mask")  
    parser.add_argument("--weightG", "-model_weight_g",type=str, default='', help="path to generator model weight")
    arg = parser.parse_args()
    return arg

def main():
    width = 4096
    height = 2048
    main_dir = _argparse().dir
    img_path = main_dir + _argparse().img
    mask_path = main_dir + _argparse().mask

    car_pano_img = preprocess_img(img_path,0.0,img_size=(width,height))
    mask_img_pano = preprocess_img(mask_path,0.0,img_size=(1000,1000))
    mask_img = preprocess_img(mask_path,0.5)

    equ = Perspective(car_pano_img)    # Load equirectangular image
    per_img = equ.GetPerspective(167, 0, -90, 128, 128)  # Specify parameters(FOV, theta, phi, height, width)

    equ = Equirectangular(mask_img_pano,167, 0, -90,'img')    
    mask_pano_img,mask = equ.GetEquirec(height,width)


    # initial generator model
    keras.backend.clear_session()
    generator = InpaintingModel().prepare_model(input_size=(128,128,3))
    generator.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[dice_coef])
    keras.utils.plot_model(generator, show_shapes=True, dpi=60, to_file='model_v2_128.png')
    if(_argparse().weightG != ''):
       generator.load_weights(_argparse().weightG) # 500

    mask = cv2.bitwise_not(mask_img) # for only u-net
    mask = mask/255.0
    # ## Mask the image
    car_img = per_img/255.0
    masked_image = mask * car_img + (1 - mask) * 1.0
    masked_image = masked_image.astype('float32')
    mask = mask.astype('float32')
    car_img = car_img.astype('float32')

    inputs = [masked_image.reshape((1,)+masked_image.shape), mask.reshape((1,)+mask.shape)]
    predict_image = generator.predict(inputs)
    inpaint_img = mask * car_img + (1 - mask) * predict_image[0]
    # print('finish predict')

    equ = Equirectangular(predict_image[0],167, 0, -90)    
    inpaint_pano_img,mask = equ.GetEquirec(height,width)
    # plt.imshow(inpaint_pano_img)

    inpaint_fill = inpaint_pano(inpaint_pano_img*255.0,mask_pano_img,car_pano_img)
    # plt.imshow(inpaint_fill)

    car_pano_gps = preprocess_img('D:/inpaint_gan/car_ds/image_test2/1_afterfill.jpg',0.0)
    equ = Equirectangular(car_pano_gps,167, 0, -90)    
    inpaint_pano_gps_img,mask = equ.GetEquirec(height,width)
    inpaint_gps_fill = inpaint_pano(inpaint_pano_gps_img,mask_pano_img,car_pano_img)

    fig, axs = plt.subplots(nrows=3, ncols=2)
    # axs[0][0].imshow(mask_pano_img/255.0)
    axs[0][0].imshow(predict_image[0])
    axs[1][0].imshow(inpaint_pano_img)
    axs[2][0].imshow(inpaint_fill)
    axs[0][1].imshow(car_pano_gps/255.0)
    axs[1][1].imshow(inpaint_pano_gps_img/255.0)
    axs[2][1].imshow(inpaint_gps_fill)
    plt.show()

if __name__ == '__main__':
    main()

