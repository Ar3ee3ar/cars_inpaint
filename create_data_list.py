import glob
import cv2
from sklearn.model_selection import train_test_split

path_input_list = []
path_output_list = []
mask_list = []
config_list = []
artifact_list = []

main_dir = "D:/inpaint_gan/"

with open(main_dir+'car_ds/data/artifact.txt') as f:
    line = f.readlines()
    for num_pic in line:
        artifact_list.append(str(num_pic.strip()))
# print(artifact_list[0])

def get_file_name(ds_path,path_list):
    for i in range(len(path_list)):
        # print(path_list[i])
        # input img process
        split_name_input = path_list[i].split('\\') # split path name
        path_list[i] = ds_path+split_name_input[1]
    return path_list

def remove_artifact(input,output,mask,config,pattern_name):
    for i in range(len(input)):
        for j in range(len(artifact_list)):
            try:
                if(artifact_list[j]+pattern_name == (input[i].split('/'))[2]):
                    input[i] = 0
                    output[i] =0
                    mask[i] = 0
                    config[i] = 0
            except:
                pass

    input = [x for x in input if x != 0]
    output = [x for x in output if x != 0]
    mask = [x for x in mask if x != 0]
    config = [x for x in config if x != 0]

    return input,output,mask,config


# 1000 x 1000
path_input = main_dir + "car_ds/image_test2/"
dir_list_input = glob.glob(path_input+"*_topview_target.jpg")
dir_list_input.sort()
path_output = main_dir + "car_ds/image_test2/"
dir_list_output = glob.glob(path_output+"*_afterfill.jpg")
dir_list_output.sort()
mask_img = 'car_ds/mask/rect_test2_mask.jpg'
# get only file_name
dir_list_input = get_file_name("car_ds/image_test2/",dir_list_input)
dir_list_output = get_file_name("car_ds/image_test2/",dir_list_output)
# mask_img = get_file_name("car_ds/mask/",[mask_img])
# append to list
path_input_list = path_input_list + dir_list_input
path_output_list = path_output_list + dir_list_output
mask_list = mask_list + ([mask_img] * len(dir_list_input))
config_list = config_list + ([str(0.0)] * len(dir_list_input))
path_input_list,path_output_list,mask_list,config_list = remove_artifact(path_input_list,path_output_list,mask_list,config_list,pattern_name='_topview_target.jpg')
x_train_1000, x_test_1000, y_train_1000, y_test_1000,mask_train_1000,mask_test_1000, config_train_1000, config_test_1000 = train_test_split(path_input_list, path_output_list,mask_list,config_list, test_size=0.33, random_state=123)


# 1000 x 1000 + crop (256)
path_input_list = []
path_output_list = []
mask_list = []
config_list = []
path_input = main_dir + "car_ds/image_center/"
dir_list_input = glob.glob(path_input+"*_topview_target.jpg")
dir_list_input.sort()
path_output = main_dir + "car_ds/image_center/"
dir_list_output = glob.glob(path_output+"*_afterfill.jpg")
dir_list_output.sort()
mask_img = 'car_ds/mask/rect_256_mask.jpg'
# get only file_name
dir_list_input = get_file_name("car_ds/image_center/",dir_list_input)
dir_list_output = get_file_name("car_ds/image_center/",dir_list_output)
# mask_img = get_file_name("car_ds/mask/",[mask_img])
# append to list
path_input_list = path_input_list + dir_list_input
path_output_list = path_output_list + dir_list_output
mask_list = mask_list + ([mask_img] * len(dir_list_input))
config_list = config_list + ([str(0.0)] * len(dir_list_input))
path_input_list,path_output_list,mask_list,config_list = remove_artifact(path_input_list,path_output_list,mask_list,config_list,pattern_name='_topview_target.jpg')
x_train_crop, x_test_crop, y_train_crop, y_test_crop,mask_train_crop,mask_test_crop, config_train_crop, config_test_crop = train_test_split(path_input_list, path_output_list,mask_list,config_list, test_size=0.33, random_state=123)

# 290 x 190
path_input_list = []
path_output_list = []
mask_list = []
config_list = []
path_input = main_dir + "car_ds/image_test/"
dir_list_input = glob.glob(path_input+"*_source_img_crop.jpg")
dir_list_input.sort()
path_output = main_dir + "car_ds/image_test/"
dir_list_output = glob.glob(path_output+"*_source_img_crop.jpg")
dir_list_output.sort()
mask_img = 'car_ds/mask/rect_relate_mask.jpg'
# get only file_name
dir_list_input = get_file_name("car_ds/image_test/",dir_list_input)
dir_list_output = get_file_name("car_ds/image_test/",dir_list_output)
# mask_img = get_file_name("car_ds/mask/",[mask_img])
# append to list
path_input_list = path_input_list + dir_list_input
path_output_list = path_output_list + dir_list_output
mask_list = mask_list + ([mask_img] * len(dir_list_input))
config_list = config_list + ([str(0.0)] * len(dir_list_input))
x_train_relate, x_test_relate, y_train_relate, y_test_relate,mask_train_relate,mask_test_relate, config_train_relate, config_test_relate = train_test_split(path_input_list, path_output_list,mask_list,config_list, test_size=0.33, random_state=123)

x_train = x_train_1000 + x_train_crop + x_train_relate
mask_train = mask_train_1000 + mask_train_crop + mask_train_relate
y_train = y_train_1000 + y_train_crop + y_train_relate
config_train = config_train_1000 + config_train_crop + config_train_relate

# x_train = x_train_crop
# mask_train = mask_train_crop
# y_train = y_train_crop
# config_train = config_train_crop

main_folder_path = 'car_ds/data_crop/'
# train data
with open(main_dir + main_folder_path + 'train/masked_img.txt', 'w') as f:
    f.write('\n'.join(x_train))
with open(main_dir + main_folder_path+'train/masks.txt', 'w') as f:
    f.write('\n'.join(mask_train))
with open(main_dir + main_folder_path+'train/output.txt', 'w') as f:
    f.write('\n'.join(y_train))
with open(main_dir + main_folder_path+'train/config_zoom.txt', 'w') as f:
    f.write('\n'.join(config_train))

x_test = x_test_1000 + x_test_crop + x_test_relate
mask_test = mask_test_1000 + mask_test_crop + mask_test_relate
y_test = y_test_1000 + y_test_crop + y_test_relate
config_test = config_test_1000 + config_test_crop + config_test_relate

# x_test = x_test_crop
# mask_test = mask_test_crop
# y_test = y_test_crop
# config_test = config_test_crop

# test data
with open(main_dir + main_folder_path+'test/masked_img.txt', 'w') as f:
    f.write('\n'.join(x_test))
with open(main_dir + main_folder_path+'test/masks.txt', 'w') as f:
    f.write('\n'.join(mask_test))
with open(main_dir + main_folder_path+'test/output.txt', 'w') as f:
    f.write('\n'.join(y_test))
with open(main_dir + main_folder_path+'test/config_zoom.txt', 'w') as f:
    f.write('\n'.join(config_test))




