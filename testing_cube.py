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
from PIL import Image

from dataset import Dataset, createAugment
# from partial_conv.generator import dice_coef, InpaintingModel
# pix2pix
import pix2pix.Generator as p2pG
import pix2pix.Discriminator as p2pD

from partial_conv.generator import dice_coef, InpaintingModel

from tools.utils import generate_images, wandb_log, view_test
from tools.loss import discriminator_loss,generator_loss
from tools.process_img import preprocess_img, crop_center, pad_images_to_same_size, resize
from tools.process_pano import Equirectangular,Perspective,inpaint_pano, panorama2cube, cube2panorama
from tools.py360convert.py360convert import c2e, e2c

tf.config.run_functions_eagerly(True)

# Description: command ไว้ทำตามคำสั่ง
def _argparse():
    # print('parsing args...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-main_dir",type=str, default='', help="parent directory that store dataset")
    parser.add_argument("--img", "-img_dir",type=str, default='', help="directory that store input") 
    parser.add_argument("--mask", "-mask_dir",type=str, default='', help="directory that store mask")  
    parser.add_argument("--weightG", "-model_weight_g",type=str, default='', help="path to generator model weight")
    parser.add_argument("--model_name",type=str, default='p2p', help="model name : p2p | pconv")
    parser.add_argument("--view",type=str, default='cube', help="model name : top | cube")
    parser.add_argument("--save", type=str, nargs='?', const='True', default="True", help="enable random mask")
    arg = parser.parse_args()
    return arg


def main():
    im_size = 256
    image_size = (im_size,im_size)
    input_model_size = [im_size,im_size,3]

    width = 512
    height = 256
    predict_cube_img = []

    main_dir = _argparse().dir
    img_path = _argparse().img
    mask_path = _argparse().mask

    save_path = main_dir+"test_img/full/"

    # load image with preprocess
    car_pano_img = preprocess_img(img_path,0.0,img_size=(width,height))
    mask_img_pano = preprocess_img(mask_path,0.0,img_size=(width,height)) # for inpaint in pano
    count = str(329)
    # save resize car
    car_pano_resize = Image.fromarray(((car_pano_img)).astype(np.uint8))
    car_pano_resize.save(save_path+count+"_car_pano.jpg")
    # save resize mask
    mask_pano_resize = Image.fromarray(((mask_img_pano)).astype(np.uint8))
    mask_pano_resize.save(save_path+count+"_mask_pano.jpg")

    if(_argparse().view == 'cube'):
        # change to cube map
        car_cube_img = e2c(car_pano_img, face_w=256, mode='bilinear', cube_format='list') # [front, right, back, left, top, bottom]
        mask_cube_img = e2c(mask_img_pano, face_w=256, mode='bilinear', cube_format='list') # [front, right, back, left, top, bottom]
    elif(_argparse().view == 'top'):
        # change to view
        equ = Perspective(car_pano_img)    # Load equirectangular image
        car_cube_img = equ.GetPerspective(167, 0, -90, 256, 256)  # Specify parameters(FOV, theta, phi, height, width)
        car_cube_img = [car_cube_img]
        equ = Perspective(mask_img_pano)    # Load equirectangular image
        mask_cube_img = equ.GetPerspective(167, 0, -90, 256, 256)  # Specify parameters(FOV, theta, phi, height, width)
        # mask_cube_img_1000 = equ.GetPerspective(167, 0, -90, 1000, 1000)  # Specify parameters(FOV, theta, phi, height, width)
        # save_path = "D:/inpaint_gan/test_img/"
        # impainted_image_im = Image.fromarray(((mask_cube_img_1000)).astype(np.uint8))
        # impainted_image_im.save(save_path+"mask_object.jpg")
        mask_cube_img = [mask_cube_img]

    # initial generator model
    keras.backend.clear_session()
    if(_argparse().model_name == 'pconv'):
      # generator = InpaintingModel().prepare_model(input_size=input_model_size)
      generator = InpaintingModel().build_pconv_unet(input_size=input_model_size,train_bn = True)
    elif(_argparse().model_name == 'p2p'):
      generator = p2pG.Generator(input_shape=input_model_size)
    # generator = InpaintingModel().prepare_model(input_size=(128,128,3))
    # generator.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[dice_coef])
    # keras.utils.plot_model(generator, show_shapes=True, dpi=60, to_file='model_v2_128.png')

    if(_argparse().weightG != ''):
    #    print(_argparse().weightG)
       generator.load_weights(_argparse().weightG) # 500

    for i in range(len(mask_cube_img)):
        mask_cube_img[i] = cv2.bitwise_not(mask_cube_img[i]) # for only u-net

    testgen = createAugment(np.array(car_cube_img), np.array(car_cube_img),np.array(mask_cube_img),batch_size=1,dim=image_size, shuffle=False, random_mask=False)
    for i in range(len(testgen)):
        [masked_image, mask], car_img = testgen[i]
        # plt.imshow(masked_image[0])
        # plt.show()

        if(_argparse().model_name == 'p2p'):
            # pix2pix ------------------------------------
            car_img = (car_img - 0.5)/0.5 # normalize [-1,1]
            masked_image = (masked_image - 0.5)/0.5 # normalize [-1,1]
            inputs = masked_image # pix2pix
            predict_image = generator(inputs, training=True)
            predict_image_norm = ((predict_image[0]* 0.5) + 0.5).numpy()# de-normalize [0,1]
            # plt.imshow(predict_image_norm)
            # plt.show()
            masked_image = (masked_image* 0.5) + 0.5 # de-normalize [0,1]
            car_img = (car_img* 0.5) + 0.5 # de-normalize [0,1]
        # ------------------------------------------------
        elif(_argparse().model_name == 'pconv'):
            inputs = [masked_image, mask] # pconv
            predict_image = generator(inputs, training=True)
            predict_image_norm = predict_image[0].numpy()
            # predict_image = generator.predict(inputs)
            # predict_image_norm = predict_image[0]
            # fig, axs = plt.subplots(nrows=2, ncols=3)
            # axs[0][0].imshow(mask_pano_img/255.0)

        predict_image_norm = (mask[0] * masked_image[0]) + ((1 - mask[0]) * predict_image_norm)
        
        predict_cube_img.append(predict_image_norm)

    if(_argparse().view == 'cube'):
        cube_img = c2e(predict_cube_img, h=height, w=width, mode='bilinear', cube_format='list') # [front, right, back, left, top, bottom]
        # cube_img = cube_img/255.0
        plt.imshow(cube_img)
        plt.show()
        if(bool(strtobool(_argparse().save))):
            save_path = main_dir+"test_img/full/"
            out_image_im = Image.fromarray(((cube_img)*255.0).astype(np.uint8))
            out_image_im.save(save_path+count+"_gan_inpaint_cube_con_output.jpg")

    # print(np.array(predict_cube_img).shape)
    if(_argparse().view == 'cube'):
        cube_img = c2e(predict_cube_img, h=height, w=width, mode='bilinear', cube_format='list') # [front, right, back, left, top, bottom]
        # cube_img = cube_img/255.0
        inpaint_fill = inpaint_pano(cube_img*255.0,mask_img_pano,car_pano_img)
        fig, axs = plt.subplots(nrows=2, ncols=2)
        axs[0][0].imshow(cube_img)
        axs[0][1].imshow(inpaint_fill)
        # plt.imshow(inpaint_fill)
        plt.show()
        if(bool(strtobool(_argparse().save))):
            save_path = main_dir+"test_img/full/"
            impainted_image_im = Image.fromarray(((inpaint_fill)*255.0).astype(np.uint8))
            impainted_image_im.save(save_path+count+"_gan_inpaint_cube_con.jpg")
    elif(_argparse().view == 'top'):
        cube_img = predict_cube_img[0]
        equ = Equirectangular(cube_img,167, 0, -90)    
        inpaint_pano_img,mask = equ.GetEquirec(height,width)
        inpaint_fill = inpaint_pano(inpaint_pano_img*255.0,mask_img_pano,car_pano_img)
        fig, axs = plt.subplots(nrows=2, ncols=3)
        axs[0][0].imshow(car_cube_img[0])
        axs[0][1].imshow(mask)
        axs[0][2].imshow(cube_img)
        axs[1][0].imshow(inpaint_pano_img)
        axs[1][1].imshow(inpaint_fill)
        plt.show()

        if(bool(strtobool(_argparse().save))):

            # inpaint only surrounding
            # name_file = name_full[4].split('_') 
            # count = str(int((name_file[2]).split('.')[0]))
            impainted_image_im = Image.fromarray(((inpaint_fill)*255.0).astype(np.uint8))
            impainted_image_im.save(save_path+count+"_gan_inpaint_uncon.jpg")


    # equ = Perspective(cube_img)    # Load equirectangular image
    # top_view = equ.GetPerspective(167, 0, -90, 256, 256)  # Specify parameters(FOV, theta, phi, height, width)

    # plt.imshow(cube_img)
    # plt.show()


if __name__ == '__main__':
    main()

