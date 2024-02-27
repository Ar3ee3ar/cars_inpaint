import glob
from PIL import Image
import numpy as np

from tools.process_img import crop_center,preprocess_img

main_dir = "D:/inpaint_gan/car_ds/"

def process_ds(file_path,save_path):
    img_pano = preprocess_img(file_path,0.0,(1000,1000)) # read file to array
    img_crop_center = crop_center(img_pano,(256,256)) # resize to 290,190
    img_crop_center = Image.fromarray(img_crop_center.astype(np.uint8)) # convert to PIL Image
    img_crop_center.save(save_path) # save image

path_input = main_dir + "image_test2/"
dir_list_input = glob.glob(path_input+"*_topview_target.jpg")
dir_list_output = glob.glob(path_input+"*_afterfill.jpg")
dir_list_input.sort()
dir_list_output.sort()
# print(dir_list_input[0].split('\\'))
for i in range(len(dir_list_input)):
    # input img process
    split_name_input = dir_list_input[i].split('\\') # split path name
    save_path = main_dir + 'image_center/' + split_name_input[len(split_name_input) - 1]
    process_ds(dir_list_input[i],save_path)
    # output img process
    split_name_input = dir_list_output[i].split('\\') # split path name
    save_path = main_dir + 'image_center/' + split_name_input[len(split_name_input) - 1]
    process_ds(dir_list_output[i],save_path)
