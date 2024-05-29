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
import time

from dataset import Dataset, createAugment
# from partial_conv.generator import dice_coef, InpaintingModel
# pix2pix
import pix2pix.Generator as p2pG
import pix2pix.Discriminator as p2pD

from partial_conv.generator import dice_coef, InpaintingModel

from tools.utils import generate_images, wandb_log, view_test
from tools.loss import discriminator_loss,generator_loss
from tools.process_img import preprocess_img, crop_center, pad_images_to_same_size, resize
from tools.process_pano import Equirectangular,Perspective,inpaint_pano

tf.config.run_functions_eagerly(True)

# Description: command ไว้ทำตามคำสั่ง
def _argparse():
    # print('parsing args...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-main_dir",type=str, default='', help="parent directory that store dataset")
    parser.add_argument('--img', '-loss-list', nargs='+', default=[], help = "number of image")
    parser.add_argument("--mask", "-mask_dir",type=str, default='', help="directory that store mask")  
    parser.add_argument("--weightG", "-model_weight_g",type=str, default='', help="path to generator model weight")
    parser.add_argument("--model_name",type=str, default='p2p', help="model name : p2p | pconv")
    parser.add_argument("--random_mask", type=str, nargs='?', const='True', default="False", help="enable random mask")
    parser.add_argument("--save", type=str, nargs='?', const='True', default="False", help="enable random mask")
    parser.add_argument("--save_fol",type=str, default='pano', help="folder name")
    arg = parser.parse_args()
    return arg

def main():
    im_size = 256
    image_size = (im_size,im_size)
    input_model_size = [im_size,im_size,3]

    width = 4096
    height = 2048
    main_dir = _argparse().dir
    count_list = _argparse().img
    mask_path = _argparse().mask
    all_time = 0.0


    # initial generator model
    keras.backend.clear_session()
    if(_argparse().model_name == 'pconv'):
      name_folder = "pconv"
      # generator = InpaintingModel().prepare_model(input_size=input_model_size)
      generator = InpaintingModel().build_pconv_unet(input_size=input_model_size,train_bn = True)
    elif(_argparse().model_name == 'p2p'):
      name_folder = "pix2pix"
      generator = p2pG.Generator(input_shape=input_model_size)
    # generator = InpaintingModel().prepare_model(input_size=(128,128,3))
    # generator.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[dice_coef])
    # keras.utils.plot_model(generator, show_shapes=True, dpi=60, to_file='model_v2_128.png')

    if(_argparse().weightG != ''):
    #    print(_argparse().weightG)
       generator.load_weights(_argparse().weightG) # 500
    
    for count_num in count_list:
      img_path = main_dir + "car_ds\pic\Input\LB_0_"+str(count_num).rjust(6, '0')+".jpg"
      # print(img_path)
      count = str(count_num)
      start_pano = time.time()
      # mask process
      mask_img_pano = preprocess_img(mask_path,img_size=(1000,1000)) # for inpaint in pano
      mask_img = preprocess_img(mask_path,img_size=(1000,1000)) # for model
      # print('preprocess: ',(time.time() - start_pano))
      # mask image
      mask_img = crop_center(mask_img,(256,256)) # resize to 290,190
      mask_img = preprocess_img(mask_img,img_size=image_size) # resize to match input model (256,256)
      # print('crop center mask: ',(time.time() - start_pano))
      # project to equirectangular (get mask image)
      equ = Equirectangular(mask_img_pano,167, 0, -90,'img')    
      mask_pano_img,mask = equ.GetEquirec(height,width)
      # save mask image
      # patch_image_im = Image.fromarray(((mask_pano_img)).astype(np.uint8))
      # patch_image_im.save("D:/inpaint_gan/car_ds/pic/car_equi_mask.jpg")
      # plt.imshow(mask_pano_img)
      # plt.show()
      # print('mask equi: ',(time.time() - start_pano))   

      # load image with preprocess
      car_pano_img = preprocess_img(img_path,img_size=(width,height))
      # if(_argparse().model_name == 'pconv'):
      # elif(_argparse().model_name =='p2p'):
      #    mask_img = preprocess_img("D:/inpaint_gan/car_ds/mask/mask_image_test.jpg",0.0,img_size=image_size) # for model

      # project to perspective (get car image)
      equ = Perspective(car_pano_img)    # Load equirectangular image
      per_img_1000 = equ.GetPerspective(167, 0, -90, 1000, 1000)  # Specify parameters(FOV, theta, phi, height, width)
      # print('perspective: ',(time.time() - start_pano))

      # plt.imshow(car_pano_img)
      # plt.show()
      # car image
      per_img = crop_center(per_img_1000,(256,256)) # resize to 290,190
      car_img = preprocess_img(per_img,img_size=image_size) # resize to match input model (256,256)
      # print('crop center car: ',(time.time() - start_pano))


      mask = cv2.bitwise_not(mask_img) # for only u-net
      mask = mask.astype('float32')
      car_img = car_img.astype('float32')

      testgen = createAugment(np.array([car_img]), np.array([car_img]),np.array([mask]),batch_size=1,dim=image_size, shuffle=False, random_mask=bool(strtobool(_argparse().random_mask)))
      [masked_image, mask], car_img = testgen[0]
      # plt.imshow(masked_image[0])
      # plt.show()

      if(_argparse().model_name == 'p2p'):
        # pix2pix ------------------------------------
        car_img = (car_img - 0.5)/0.5 # normalize [-1,1]
        masked_image = (masked_image - 0.5)/0.5 # normalize [-1,1]
        inputs = masked_image # pix2pix
        start_model = time.time()
        predict_image = generator(inputs, training=True)
        end_model = time.time()
        # print('model inpaint time: ',(end_model - start_model))
        predict_image_norm = ((predict_image[0]* 0.5) + 0.5).numpy()# de-normalize [0,1]
        # plt.imshow(predict_image_norm)
        # plt.show()
        masked_image = (masked_image* 0.5) + 0.5 # de-normalize [0,1]
        car_img = (car_img* 0.5) + 0.5 # de-normalize [0,1]
        # ------------------------------------------------
      elif(_argparse().model_name == 'pconv'):
        inputs = [masked_image, mask] # pconv
        start_model = time.time()
        predict_image = generator(inputs, training=True)
        end_model = time.time()
        # print('model inpaint time: ',(end_model - start_model))
        predict_image_norm = predict_image[0].numpy()
        # save patch image
        pred_image_im = Image.fromarray(((predict_image_norm)*255.0).astype(np.uint8))
        pred_image_im.save(main_dir+"test_img/"+name_folder+"/"+_argparse().save_fol+"/"+count+"_pred.jpg")
      # print('finish predict')

      # print(predict_image_norm)
      # plt.imshow(predict_image_norm)
      # plt.show()
      # print(predict_image_norm.shape)
      mask_img_pano = mask_img_pano/255.0
      per_img_1000 = per_img_1000/255.0
      # if(_argparse().random_mask == False):
      predict_image_norm = preprocess_img(predict_image_norm,img_size=(256,256)) # resize to match input model (256,256)
      predict_image_norm = pad_images_to_same_size(predict_image_norm)
      predict_image_norm = ((1 - mask_img_pano) * per_img_1000) + (mask_img_pano * predict_image_norm)
      # print('inpaint output: ',(time.time() - start_pano))

      # plt.imshow(predict_image_norm)
      # plt.show()

      # print(predict_image_norm.shape)

      equ = Equirectangular(predict_image_norm,167, 0, -90)    
      inpaint_pano_img,mask = equ.GetEquirec(height,width)
      # plt.imshow(inpaint_pano_img)

      inpaint_fill = inpaint_pano(inpaint_pano_img*255.0,mask_pano_img,car_pano_img)
      # mask_pano_bitwise = mask_pano_img/255.0
      # mask_pano_bitwise = cv2.bitwise_not(mask_pano_bitwise) # for only u-net
      inpaint_mask = inpaint_pano(mask_pano_img,mask_pano_img,car_pano_img)
      # print('inpaint pano: ',(time.time() - start_pano))
      end_pano = time.time()
      all_time = all_time + (end_pano - start_pano)
      if(int(count) % 10 == 0):
         print(count,' : ',all_time/float(count))
      # print("all process time: ",(end_pano - start_pano))
      # plt.imshow(inpaint_fill)
      car_pano_gps = preprocess_img(main_dir+'car_ds/pic/image_test2/'+count+'_afterfill.jpg',(1000,1000))
      equ = Equirectangular(car_pano_gps,167, 0, -90)    
      inpaint_pano_gps_img,mask = equ.GetEquirec(height,width)
      inpaint_gps_fill = inpaint_pano(inpaint_pano_gps_img,mask_pano_img,car_pano_img)

      # fig, axs = plt.subplots(nrows=3, ncols=2)
      # # axs[0][0].imshow(mask_pano_img/255.0)
      # axs[0][0].imshow(inpaint_mask)
      # axs[1][0].imshow(predict_image_norm)
      # axs[2][0].imshow(inpaint_fill)
      # axs[0][1].imshow(inpaint_mask)
      # axs[1][1].imshow(car_pano_gps/255.0)
      # axs[2][1].imshow(inpaint_gps_fill)
      # plt.show()

      # save = False
      if(bool(strtobool(_argparse().save))):
        save_path = main_dir+"test_img/"+name_folder+"/"+_argparse().save_fol+"/"
        # save input image
        input_image_im = Image.fromarray(((inpaint_mask)*255.0).astype(np.uint8))
        input_image_im.save(save_path+count+"_gan_input.jpg")
        # save inpaint image
        impainted_image_im = Image.fromarray(((inpaint_fill)*255.0).astype(np.uint8))
        impainted_image_im.save(save_path+count+"_gan_inpaint.jpg")
        # save patch image
        patch_image_im = Image.fromarray(((inpaint_gps_fill)*255.0).astype(np.uint8))
        patch_image_im.save(save_path+count+"_patch_inpaint.jpg")

if __name__ == '__main__':
    main()

