from box import Box

config = {
    "dir": "D:/inpaint_gan/",
    "mask" : "D:/inpaint_gan/car_ds/pic/mask/binary_mask_test.jpg",
    "weightG" : "D:/inpaint_gan/weight/nakorn/p286_new/bestG_weight.h5",
    "model_name": "p2p",
    "ds_name": "c1",
    "random_box": False,
    "random_mask": False,
    "save": True,
    "save_fol": "nakorn/p286_new/best",
    "img": [2060],
    "pic_4x" : ""
    }

cfg_test = Box(config)