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
# from config_test import cfg_test
from config_test_colab import cfg_test
# from partial_conv.generator import dice_coef, InpaintingModel
# pix2pix
import pix2pix.Generator as p2pG
import pix2pix.Discriminator as p2pD

# from partial_conv.generator import dice_coef, InpaintingModel

from tools.utils import generate_images, wandb_log, view_test
from tools.loss import discriminator_loss,generator_loss
from tools.process_img import preprocess_img, crop_center, pad_images_to_same_size, resize, move_img
from tools.process_pano import Equirectangular,Perspective,inpaint_pano
# from tools.process_img import preprocess_img, move_img, flip_img, zoom_img, crop_center

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
    parser.add_argument("--random_box", type=str, nargs='?', const='True', default="False", help="enable random mask")
    parser.add_argument("--random_mask", type=str, nargs='?', const='True', default="False", help="enable random mask")
    parser.add_argument("--save", type=str, nargs='?', const='True', default="False", help="enable random mask")
    parser.add_argument("--save_fol",type=str, default='pano', help="folder name")
    parser.add_argument("--pic_4x",type=str, default='', help="folder name")
    arg = parser.parse_args()
    return arg

def create_fol(path):
    if not os.path.exists(path): 
        os.makedirs(path)
    else:
        print('already create '+path)

def seamless(source,target,mask_img):
    mask = mask_img
    # mask = cv2.imread(mask_path)
    # mask = cv2.resize(mask,(2000,2000))
    # mask[mask != 0] = 255
    center = (499,450)
    # center = (1000,910)
    print(center)
    im_clone = cv2.seamlessClone(source, target, mask, center, cv2.NORMAL_CLONE)

    # cv2.imshow("img",cv2.resize(mask,(256,256)))
    # cv2.imshow("result",cv2.resize(im_clone,(256,256)))
    # cv2.waitKey() 
    # cv2.destroyAllWindows()
    return im_clone

def main(cfg_test):
    im_size = 512
    image_size = (im_size,im_size)
    input_model_size = [im_size,im_size,3]

    width = 4096
    height = 2048
    main_dir = cfg_test.dir
    count_list = cfg_test.img
    mask_path = cfg_test.mask
    all_time = 0.0

    coor = {
      "fov": 167,
      "x": 27,
      "y": -90
    }


    # initial generator model
    keras.backend.clear_session()
    # if(cfg_test.model_name == 'pconv'):
    #   name_folder = "pconv"
    #   # generator = InpaintingModel().prepare_model(input_size=input_model_size)
    #   generator = InpaintingModel().build_pconv_unet(input_size=input_model_size,train_bn = True)
    if(cfg_test.model_name == 'p2p'):
      name_folder = "pix2pix"
      generator = p2pG.Generator(input_shape=input_model_size)
    # generator = InpaintingModel().prepare_model(input_size=(128,128,3))
    # generator.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[dice_coef])
    # keras.utils.plot_model(generator, show_shapes=True, dpi=60, to_file='model_v2_128.png')

    if(cfg_test.weightG != ''):
    #    print(cfg_test.weightG)
       generator.load_weights(cfg_test.weightG) # 500
       
    create_fol(main_dir+"test_img/"+name_folder)
    create_fol(main_dir+"test_img/"+name_folder+"/"+cfg_test.save_fol+"/"+cfg_test.ds_name)
    
    for count_num in count_list:
      if(cfg_test.ds_name == "bridge"):
        img_path = main_dir + "car_ds\pic\Input\LB_0_"+str(count_num).rjust(6, '0')+".jpg" # 
        coor["x"] = 0
      elif(cfg_test.ds_name == "8k"):
        img_path ="D:/new_car_ds/pano_8k/pano_8k/img_"+str(count_num).rjust(6, '0')+".jpg" #"D:/inpaint_gan/test_img/pix2pix/new_512/26082024/140/"+str(count_num)+"_gan_inpaint.jpg" 
        coor["x"] = 27
      elif(cfg_test.ds_name == "c1"):
        img_path ="D:/new_car_ds/11-01/C1/Camera/img_"+str(count_num).rjust(6, '0')+".jpg" #"D:/inpaint_gan/test_img/pix2pix/new_512/26082024/140/"+str(count_num)+"_gan_inpaint.jpg" 
        coor["x"] = 27
      elif(cfg_test.ds_name == "c2"):
        img_path ="D:/new_car_ds/11-01/C2/Camera/img_"+str(count_num).rjust(6, '0')+".jpg" #"D:/inpaint_gan/test_img/pix2pix/new_512/26082024/140/"+str(count_num)+"_gan_inpaint.jpg" 
        coor["x"] = 27
      elif(cfg_test.ds_name == "i9-1"):
        img_path ="D:/new_car_ds/11-01/I9-1/Camera/img_"+str(count_num).rjust(6, '0')+".jpg" #"D:/inpaint_gan/test_img/pix2pix/new_512/26082024/140/"+str(count_num)+"_gan_inpaint.jpg" 
        coor["x"] = 27
      elif(cfg_test.ds_name == "i9-10"):
        img_path ="D:/new_car_ds/11-01/I9-10/Camera/img_"+str(count_num).rjust(6, '0')+".jpg" #"D:/inpaint_gan/test_img/pix2pix/new_512/26082024/140/"+str(count_num)+"_gan_inpaint.jpg" 
        coor["x"] = 27
      elif(cfg_test.ds_name == "upload"):
        img_path = str(count_num)
        count_num = (img_path.split('/')[-1]).split('.')[0]
        coor["x"] = 27
      # print(img_path)
      count = str(count_num)
      start_pano = time.time()
      # mask process
      mask_img_pano = preprocess_img(mask_path,img_size=(1000,1000)) # for inpaint in pano
      # plt.imshow(mask_img_pano)
      # plt.show()
      # mask_img_pano[mask_img_pano <255] = 0
      mask_img = preprocess_img(mask_path,img_size=(1000,1000)) # for model
      # plt.imshow(mask_img)
      # plt.show()
      # mask_img[mask_img <255] = 0
      # print('preprocess: ',(time.time() - start_pano))
      # mask image
      mask_img = crop_center(mask_img,(512,512)) # resize to 290,190
      # mask_img = preprocess_img(mask_img,img_size=image_size) # resize to match input model (256,256)
      # print('crop center mask: ',(time.time() - start_pano))
      # project to equirectangular (get mask image)
      equ = Equirectangular(mask_img_pano,coor["fov"],coor["x"],coor["y"],'img')    
      mask_pano_img,_ = equ.GetEquirec(height,width)
      # save mask image
      # patch_image_im = Image.fromarray(((mask_pano_img)).astype(np.uint8))
      # patch_image_im.save("D:/inpaint_gan/car_ds/pic/car_equi_mask.jpg")
      # plt.imshow(mask_pano_img)
      # plt.show()
      # print('mask equi: ',(time.time() - start_pano))   

      # load image with preprocess
      car_pano_img = preprocess_img(img_path,img_size=(width,height))
      # if(cfg_test.model_name == 'pconv'):
      # elif(cfg_test.model_name =='p2p'):
      #    mask_img = preprocess_img("D:/inpaint_gan/car_ds/mask/mask_image_test.jpg",0.0,img_size=image_size) # for model

      # project to perspective (get car image)
      equ = Perspective(car_pano_img)    # Load equirectangular image
      per_img_1000 = equ.GetPerspective(coor["fov"],coor["x"],coor["y"], 1000, 1000)  # Specify parameters(FOV, theta, phi, height, width)
      # print('perspective: ',(time.time() - start_pano))

      # plt.imshow(car_pano_img)
      # plt.show()
      # car image
      per_img = crop_center(per_img_1000,(512,512)) # resize to 290,190
      car_img = preprocess_img(per_img,img_size=image_size) # resize to match input model (256,256)
      # print('crop center car: ',(time.time() - start_pano))

      # plt.imshow(mask_img)
      # plt.show()
      mask = cv2.bitwise_not(mask_img) # for on\ly u-net
      # mask[mask>0]=1
      mask = mask.astype('float32')
      car_img = car_img.astype('float32')

      testgen = createAugment(np.array([car_img]), np.array([car_img]),np.array([mask]),batch_size=1,dim=image_size, shuffle=False, random_box=cfg_test.random_box, random_mask=cfg_test.random_mask, training = False)
      [masked_image, mask], car_img = testgen[0]
      # plt.imshow(masked_image[0])
      # plt.show()

      if(cfg_test.model_name == 'p2p'):
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
      elif(cfg_test.model_name == 'pconv'):
        inputs = [masked_image, mask] # pconv
        start_model = time.time()
        predict_image = generator(inputs, training=True)
        end_model = time.time()
        # print('model inpaint time: ',(end_model - start_model))
        predict_image_norm = predict_image[0].numpy()
        # save patch image
        pred_image_im = Image.fromarray(((predict_image_norm)*255.0).astype(np.uint8))
        pred_image_im.save(main_dir+"test_img/"+name_folder+"/"+cfg_test.save_fol+"/"+cfg_test.ds_name+"/"+count+"_pred.jpg")
      # print('finish predict')

      # print(predict_image_norm)
      # plt.imshow(predict_image_norm)
      # plt.show()
      # print(predict_image_norm.shape)
      mask_img_pano = mask_img_pano/255.0
      per_img_1000 = per_img_1000/255.0
      # if(cfg_test.random_mask == False):
      # predict_image_norm = preprocess_img(predict_image_norm,img_size=(512,512)) # resize to match input model (256,256)
      if(cfg_test.save):
        save_path = main_dir+"test_img/"+name_folder+"/"+cfg_test.save_fol+"/"+cfg_test.ds_name+"/"
        pred_image_im = Image.fromarray(((predict_image_norm)*255.0).astype(np.uint8))
        pred_image_im.save(save_path+count+"_gan_pred.jpg")
        cv2.imwrite(save_path+count+"_gan_pred_cv.jpg", cv2.cvtColor(predict_image_norm*255.0, cv2.COLOR_RGB2BGR))
        inpaint_img = mask * car_img + (1 - mask) * predict_image_norm
        # plt.imshow(mask[0])
        # plt.show()
        # plt.imshow((mask * car_img)[0])
        # plt.show()
        # plt.imshow(((1 - mask) * predict_image_norm)[0])
        # plt.show()
        over_image_im = Image.fromarray(((inpaint_img[0])*255.0).astype(np.uint8))
        over_image_im.save(save_path+count+"_gan_over.jpg")
      predict_image_norm = pad_images_to_same_size(predict_image_norm)
      # normal inpaint -------------------
      # predict_image_norm = ((1 - mask_img_pano) * per_img_1000) + (mask_img_pano * predict_image_norm)
      # # seamless ----------------------
      mask_path_seam = "mask_colab/8k_focus_simple2.jpg"

      # project mask image
      mask_img_seam = cv2.resize(cv2.imread(mask_path_seam),(1000,1000)) # for model

      # project to equirectangular (get mask image)
      # equ_seam = Equirectangular(mask_img_seam, 167, 27, -90,'img')    
      # mask_pano_img_seam,_ = equ_seam.GetEquirec(height,width) 
      # mask_pano_img_seam = mask_pano_img_seam.astype(np.uint8)
      print(type(predict_image_norm),' ',predict_image_norm.shape)
      print(type(per_img_1000),' ',per_img_1000.shape)
      print(type(mask_img_seam),' ',mask_img_seam.shape)
      predict_image_norm = seamless((predict_image_norm*255.0).astype(np.uint8),(per_img_1000*255.0).astype(np.uint8),mask_img_seam)
      # predict_image_norm = cv2.cvtColor(np.array(predict_image_norm), cv2.COLOR_BGR2RGB)
      predict_image_norm = predict_image_norm/255.0
      # ---------------------------------------
      if(cfg_test.save):
        save_path = main_dir+"test_img/"+name_folder+"/"+cfg_test.save_fol+"/"+cfg_test.ds_name+"/"
        pred_image_im = Image.fromarray(((predict_image_norm)*255.0).astype(np.uint8))
        pred_image_im.save(save_path+count+"_gan_inpaint_per.jpg")
      # print('inpaint output: ',(time.time() - start_pano))

      # plt.imshow(predict_image_norm)
      # plt.show()

      # print(predict_image_norm.shape)

      if( cfg_test.pic_4x != ''):
        predict_image_norm = preprocess_img(cfg_test.pic_4x,img_size=(4000,4000))
        predict_image_norm = predict_image_norm/255.0

      equ = Equirectangular(predict_image_norm,coor["fov"],coor["x"],coor["y"])    
      inpaint_pano_img,_ = equ.GetEquirec(height,width)
      # plt.imshow(inpaint_pano_img)

      inpaint_fill = inpaint_pano(inpaint_pano_img*255.0,mask_pano_img,car_pano_img)
      # mask_pano_bitwise = mask_pano_img/255.0
      # mask_pano_bitwise = cv2.bitwise_not(mask_pano_bitwise) # for only u-net
      inpaint_mask = inpaint_pano(mask_pano_img,mask_pano_img,car_pano_img)
      # print('inpaint pano: ',(time.time() - start_pano))
      end_pano = time.time()
      all_time = all_time + (end_pano - start_pano)
      # if(int(count) % 10 == 0):
      #    print(count,' : ',all_time/float(count))
      # print("all process time: ",(end_pano - start_pano))
      # plt.imshow(inpaint_fill)
      # car_pano_gps = preprocess_img(main_dir+'car_ds/pic/image_test2/'+count+'_afterfill.jpg',(1000,1000))
      # equ = Equirectangular(car_pano_gps,coor["fov"],coor["x"],coor["y"])    
      # inpaint_pano_gps_img,mask = equ.GetEquirec(height,width)
      # inpaint_gps_fill = inpaint_pano(inpaint_pano_gps_img,mask_pano_img,car_pano_img)

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
      if(cfg_test.save):
        save_path = main_dir+"test_img/"+name_folder+"/"+cfg_test.save_fol+"/"+cfg_test.ds_name+"/"
        # save input image per
        input_image_per = Image.fromarray(((car_img[0])*255.0).astype(np.uint8))
        input_image_per.save(save_path+count+"_gan_input_per.jpg")
        # save input image per
        input_mimage_per = Image.fromarray(((masked_image[0])*255.0).astype(np.uint8))
        input_mimage_per.save(save_path+count+"_gan_masked_input_per.jpg")
        # save input image
        input_image_im = Image.fromarray(((inpaint_mask)*255.0).astype(np.uint8))
        input_image_im.save(save_path+count+"_gan_input.jpg")
        # save inpaint image
        impainted_image_im = Image.fromarray(((inpaint_fill)*255.0).astype(np.uint8))
        impainted_image_im.save(save_path+count+"_gan_inpaint.jpg")
        # save patch image
        # patch_image_im = Image.fromarray(((inpaint_gps_fill)*255.0).astype(np.uint8))
        # patch_image_im.save(save_path+count+"_patch_inpaint.jpg")

if __name__ == '__main__':
    main(cfg_test)

