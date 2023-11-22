import glob
import cv2
from sklearn.model_selection import train_test_split

path_input_list = []
path_output_list = []
mask_list = []
config_list = []

main_dir = "/content/drive/MyDrive/"

# 1000 x 1000
path_input = main_dir + "car_ds/image_test2/"
dir_list_input = glob.glob(path_input+"*_topview_target.jpg")
dir_list_input.sort()
path_output = main_dir + "car_ds/image_test2/"
dir_list_output = glob.glob(path_output+"*_afterfill.jpg")
dir_list_output.sort()
mask_img = main_dir + 'car_ds/mask/mask_image_test2.jpg'
# append to list
path_input_list = path_input_list + dir_list_input
path_output_list = path_output_list + dir_list_output
mask_list = mask_list + ([mask_img] * len(dir_list_input))
config_list = config_list + ([str(0)] * len(dir_list_input))


# 1000 x 1000 + crop
path_input = main_dir + "car_ds/image_test2/"
dir_list_input = glob.glob(path_input+"*_topview_target.jpg")
dir_list_input.sort()
path_output = main_dir + "car_ds/image_test2/"
dir_list_output = glob.glob(path_output+"*_afterfill.jpg")
dir_list_output.sort()
mask_img = main_dir + 'car_ds/mask/mask_image_test2.jpg'
# append to list
path_input_list = path_input_list + dir_list_input
path_output_list = path_output_list + dir_list_output
mask_list = mask_list + ([mask_img] * len(dir_list_input))
config_list = config_list + ([str(0.5)] * len(dir_list_input))

# 290 x 190
path_input = main_dir + "car_ds/image_test/"
dir_list_input = glob.glob(path_input+"*_target_img_crop.jpg")
dir_list_input.sort()
path_output = main_dir + "car_ds/image_test/"
dir_list_output = glob.glob(path_output+"*_afterfill_crop.jpg")
dir_list_output.sort()
mask_img = main_dir + 'car_ds/mask/mask_image_test.jpg'
# append to list
path_input_list = path_input_list + dir_list_input
for i in range(len(path_input_list)):
    path_input_list[i] = (path_input_list[i].split(main_dir))[1]
path_output_list = path_output_list + dir_list_output
for i in range(len(path_output_list)):
    path_output_list[i] = (path_output_list[i].split(main_dir))[1]
mask_list = mask_list + ([mask_img] * len(dir_list_input))
for i in range(len(mask_list)):
    mask_list[i] = (mask_list[i].split(main_dir))[1]
config_list = config_list + ([str(0)] * len(dir_list_input))

x_train, x_test, y_train, y_test,mask_train,mask_test, config_train, config_test = train_test_split(path_input_list, path_output_list,mask_list,config_list, test_size=0.33, random_state=123)

# train data
with open(main_dir + 'car_ds/data/train/masked_img.txt', 'w') as f:
    f.write('\n'.join(x_train))
with open(main_dir + 'car_ds/data/train/masks.txt', 'w') as f:
    f.write('\n'.join(mask_train))
with open(main_dir + 'car_ds/data/train/output.txt', 'w') as f:
    f.write('\n'.join(y_train))
with open(main_dir + 'car_ds/data/train/config_zoom.txt', 'w') as f:
    f.write('\n'.join(config_train))

# test data
with open(main_dir + 'car_ds/data/test/masked_img.txt', 'w') as f:
    f.write('\n'.join(x_test))
with open(main_dir + 'car_ds/data/test/masks.txt', 'w') as f:
    f.write('\n'.join(mask_test))
with open(main_dir + 'car_ds/data/test/output.txt', 'w') as f:
    f.write('\n'.join(y_test))
with open(main_dir + 'car_ds/data/test/config_zoom.txt', 'w') as f:
    f.write('\n'.join(config_test))


