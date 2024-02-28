import argparse
import glob
import cv2
from sklearn.model_selection import train_test_split

main_dir = "D:/inpaint_gan/"

# Description: command ไว้ทำตามคำสั่ง
def _argparse():
    # print('parsing args...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",type=str, default='', help="create to create per|cube dataset")
    parser.add_argument("--config_name", "-config_name",type=str, default='', help="folder name to store config dataset")
    arg = parser.parse_args()
    return arg

def get_file_name(ds_path,path_list):
    for i in range(len(path_list)):
        # print(path_list[i])
        # input img process
        split_name_input = path_list[i].split('\\') # split path name
        path_list[i] = ds_path+split_name_input[1]
    return path_list

def remove_artifact(input,output,mask=None,config=None,pattern_name='',artifact_list=[]):
    print(artifact_list[0]+pattern_name,' == ',(input[0].split('/'))[3])
    for i in range(len(input)):
        for j in range(len(artifact_list)):
            try:
                if(artifact_list[j]+pattern_name == (input[i].split('/'))[3]):
                    input[i] = 0
                    output[i] =0
                    if mask is not None:
                        mask[i] = 0
                    if config is not None:
                        config[i] = 0
            except:
                pass

    input = [x for x in input if x != 0]
    output = [x for x in output if x != 0]
    if mask is not None:
        mask = [x for x in mask if x != 0]
    if config is not None:
        config = [x for x in config if x != 0]

    if (mask is not None) and (config is not None):
        return input,output,mask,config
    else:
        if (mask is None) and (config is None):
            return input,output
        elif mask is None:
            return input, output, config
        elif config is None:
            return input, output, mask

def cat_cube_face(cube_list, face):
    name_cube_list = []
    for i in range(len(cube_list)):
        name_pic = (cube_list[i].split('/'))[-1]
        # print('name_pic: ',name_pic,' <- ',cube_list[i])
        cube_face = (name_pic.split('_'))[-1].split('.')
        # print('cube_face: ',cube_face)
        if(cube_face[0] == face):
           name_cube_list.append(cube_list[i])
    
    return name_cube_list


def perspective_ds(data_file_name):
    path_input_list = []
    path_output_list = []
    mask_list = []
    config_list = []
    artifact_list = []
    with open(main_dir+'car_ds/train_test_config/'+data_file_name+'/artifact.txt') as f:
        line = f.readlines()
        for num_pic in line:
            artifact_list.append(str(num_pic.strip()))
    # print(artifact_list[0])

    # 1000 x 1000
    path_input = main_dir + "car_ds/pic/image_test2/"
    dir_list_input = glob.glob(path_input+"*_topview_target.jpg")
    dir_list_input.sort()
    path_output = main_dir + "car_ds/pic/image_test2/"
    dir_list_output = glob.glob(path_output+"*_afterfill.jpg")
    dir_list_output.sort()
    mask_img = 'car_ds/pic/mask/rect_test2_mask.jpg'
    # get only file_name
    dir_list_input = get_file_name("car_ds/pic/image_test2/",dir_list_input)
    dir_list_output = get_file_name("car_ds/pic/image_test2/",dir_list_output)
    # mask_img = get_file_name("car_ds/mask/",[mask_img])
    # append to list
    path_input_list = path_input_list + dir_list_input
    path_output_list = path_output_list + dir_list_output
    mask_list = mask_list + ([mask_img] * len(dir_list_input))
    config_list = config_list + ([str(0.0)] * len(dir_list_input))
    path_input_list,path_output_list,mask_list,config_list = remove_artifact(path_input_list,path_output_list,mask_list,config_list,pattern_name='_topview_target.jpg',artifact_list = artifact_list)
    x_train_1000, x_test_1000, y_train_1000, y_test_1000,mask_train_1000,mask_test_1000, config_train_1000, config_test_1000 = train_test_split(path_input_list, path_output_list,mask_list,config_list, test_size=0.33, random_state=123)


    # 1000 x 1000 + crop (256)
    path_input_list = []
    path_output_list = []
    mask_list = []
    config_list = []
    path_input = main_dir + "car_ds/pic/image_center/"
    dir_list_input = glob.glob(path_input+"*_topview_target.jpg")
    dir_list_input.sort()
    path_output = main_dir + "car_ds/pic/image_center/"
    dir_list_output = glob.glob(path_output+"*_afterfill.jpg")
    dir_list_output.sort()
    mask_img = 'car_ds/pic/mask/rect_256_mask.jpg'
    # get only file_name
    dir_list_input = get_file_name("car_ds/pic/image_center/",dir_list_input)
    dir_list_output = get_file_name("car_ds/pic/image_center/",dir_list_output)
    # mask_img = get_file_name("car_ds/mask/",[mask_img])
    # append to list
    path_input_list = path_input_list + dir_list_input
    path_output_list = path_output_list + dir_list_output
    mask_list = mask_list + ([mask_img] * len(dir_list_input))
    config_list = config_list + ([str(0.0)] * len(dir_list_input))
    path_input_list,path_output_list,mask_list,config_list = remove_artifact(path_input_list,path_output_list,mask_list,config_list,pattern_name='_topview_target.jpg',artifact_list = artifact_list)
    x_train_crop, x_test_crop, y_train_crop, y_test_crop,mask_train_crop,mask_test_crop, config_train_crop, config_test_crop = train_test_split(path_input_list, path_output_list,mask_list,config_list, test_size=0.33, random_state=123)

    # 290 x 190
    path_input_list = []
    path_output_list = []
    mask_list = []
    config_list = []
    path_input = main_dir + "car_ds/pic/image_test/"
    dir_list_input = glob.glob(path_input+"*_source_img_crop.jpg")
    dir_list_input.sort()
    path_output = main_dir + "car_ds/pic/image_test/"
    dir_list_output = glob.glob(path_output+"*_source_img_crop.jpg")
    dir_list_output.sort()
    mask_img = 'car_ds/pic/mask/rect_relate_mask.jpg'
    # get only file_name
    dir_list_input = get_file_name("car_ds/pic/image_test/",dir_list_input)
    dir_list_output = get_file_name("car_ds/pic/image_test/",dir_list_output)
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

    main_folder_path = 'car_ds/train_test_config/'+data_file_name+'/'
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


def cube_ds(data_file_name):
    # 'LB_0_000001_B.jpg'
    artifact_per_list = []
    artifact_cube_list = []
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    with open(main_dir+'car_ds/train_test_config/'+data_file_name+'/artifact_per/artifact.txt') as f:
        line = f.readlines()
        for num_pic in line:
            artifact_per_list.append(str(num_pic.strip()))
    face_list = ['F', 'B', 'L', 'R']
    for face in face_list:
        with open(main_dir+'car_ds/train_test_config/'+data_file_name+'/artifact_cube/'+face+'.txt') as f:
            line = f.readlines()
            for num_pic in line:
                name_pic = "LB_0_"+str(num_pic.strip()).rjust(6, '0')+"_"+face+".jpg"
                artifact_cube_list.append(name_pic)

        # 1000 x 1000 + crop (256)
        path_input_list = []
        path_output_list = []
        mask_list = []
        config_list = []
        path_input = main_dir + "car_ds/pic/Output_cube/"
        dir_list_input = glob.glob(path_input+"LB*.jpg")

        dir_list_input.sort()
        # dir_list_output = dir_list_input[:] # clone input list
        # get only file_name
        dir_list_input = get_file_name("car_ds/pic/Output_cube/",dir_list_input)
        path_input_list = cat_cube_face(dir_list_input,face)
        path_output_list = path_input_list[:]
        # remove artifact
        print('data ',face,': ',len(path_input_list))
        path_input_list,path_output_list = remove_artifact(path_input_list,path_output_list,mask=None,config=None,pattern_name='',artifact_list = artifact_cube_list)
        # report result
        print('remove artifact data ',face,': ',len(path_input_list))
        # train test split
        x_train_per, x_test_per, y_train_per, y_test_per = train_test_split(path_input_list, path_output_list, test_size=0.33, random_state=123)
        # append to list
        x_train = x_train + x_train_per
        y_train = y_train + y_train_per
        x_test = x_test + x_test_per
        y_test = y_test + y_test_per

    # 1000 x 1000
    path_input = main_dir + "car_ds/pic/image_test2/"
    dir_list_input = glob.glob(path_input+"*_afterfill.jpg")
    dir_list_input.sort()
    # mask_img = 'car_ds/pic/mask/rect_test2_mask.jpg'
    # get only file_name
    path_input_list = get_file_name("car_ds/pic/image_test2/",dir_list_input)
    path_output_list = dir_list_input[:]
    # mask_img = get_file_name("car_ds/mask/",[mask_img])
    # remove artifact
    path_input_list,path_output_list = remove_artifact(path_input_list,path_output_list,pattern_name='_afterfill.jpg',artifact_list = artifact_per_list)
    # train test split
    x_train_1000, x_test_1000, y_train_1000, y_test_1000 = train_test_split(path_input_list, path_output_list, test_size=0.33, random_state=123)
    # append to list
    x_train = x_train + x_train_1000
    y_train = y_train + y_train_1000
    x_test = x_test + x_test_1000
    y_test = y_test + y_test_1000

    # report result
    print('data per: ',len(path_input_list))
    print('train data: ',len(x_train))
    print('test data: ',len(x_test))

    main_folder_path = 'car_ds/train_test_config/'+data_file_name+'/'
    # train data
    with open(main_dir + main_folder_path + 'train/masked_img.txt', 'w') as f:
        f.write('\n'.join(x_train))
    with open(main_dir + main_folder_path+'train/output.txt', 'w') as f:
        f.write('\n'.join(y_train))
    # test data
    with open(main_dir + main_folder_path + 'test/masked_img.txt', 'w') as f:
        f.write('\n'.join(x_test))
    with open(main_dir + main_folder_path+'test/output.txt', 'w') as f:
        f.write('\n'.join(y_test))

def main():
    print('mode: ',_argparse().mode)
    print('name: ',_argparse().config_name)
    mode = _argparse().mode
    folder_name = _argparse().config_name
    if(mode == 'per'):
        perspective_ds(folder_name)
    elif(mode == 'cube'):
        cube_ds(folder_name)
        


if __name__ == '__main__':
    main()



