from box import Box

config = {
    "dir": "/content/cars_inpaint/",
    "mask" : "/content/cars_inpaint/mask_colab/binary_mask_test.jpg",
    "weightG" : "/content/cars_inpaint/weight_colab/bestG_weight.h5",
    "model_name": "p2p",
    "ds_name": "upload",
    "random_box": False,
    "random_mask": False,
    "save": True,
    "save_fol": "test_show",
    "img": "/content/cars_inpaint/upload_img_colab",
    "pic_4x" : ""
    }

cfg_test = Box(config)