import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm


from tools.py360convert.py360convert import c2e, e2c, utils
from tools.process_img import preprocess_img

width = 2048
height = 1024
main_dir = 'D:/inpaint_gan/'
save_path = main_dir + 'car_ds/Output_cube/'

for count_num in tqdm(range(1,1970)):
    # define image path
    img = main_dir + "car_ds/Output/LB_0_"+str(count_num).rjust(6, '0')+".jpg"
    name_img = (img.split('/'))[-1].split('.')
    # read image
    img_cv = preprocess_img(img,0.0,img_size=(width,height))
    # project equirectangular into cubemap
    out = e2c(img_cv, face_w=1000, mode='bilinear', cube_format='dict')
    cube_face = ['F', 'R', 'B', 'L']
    # save image
    for i in range(len(cube_face)):
        out_image_im = Image.fromarray(((out[cube_face[i]])).astype(np.uint8))
        out_image_im.save(save_path + name_img[0] +'_'+ cube_face[i] +'.'+ name_img[1])
