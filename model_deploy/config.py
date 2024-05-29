from box import Box

config = {
    "dir": "D:/inpaint_gan/",
    "config_file": "per_file",
    "model": "p2p",
    "wandb" : False,
    "ckpt_path" : "",
    "save" : 5,
    "step" : 5,
    "lrG" : 1e-5,
    "lrD" : 1e-5,
    "img_size" : 256,
    "batch_size" : 1,
    "bn" : True,
    "random_mask" : True,
    "random_car" : True,
    "loss_type" : "pconv",
    "inpaint_mode" : "per",
    "train_phase" : 2,
    "cont": False,
    "adapt_weight" : "SoftAdapt",
    "update_epoch" : 2,
    "loss_lambda":{
        "LAMBDA_adv": 1,
        "LAMBDA_l1": 100,
        "LAMBDA_perc": 0.05,
        "LAMBDA_tv": 0.1,
        "LAMBDA_hole": 0,
        "LAMBDA_valid": 0,
        "LAMBDA_style": 120
    },
    }

cfg = Box(config)