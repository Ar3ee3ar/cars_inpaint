from box import Box

config = {
    "dir": "D:/inpaint_gan/",
    "config_file": "per_file",
    "model": "p2p",
    "wandb" : False,
    "ckpt_path" : "",#"D:/inpaint_gan/logs/fit/20240823-200429/training_checkpoints/",
    "save" : 3,
    "step" : 3,
    "lrG" : 1e-5,
    "lrD" : 1e-5,
    "img_size" : 512,
    "crop_size" : 256,
    "crop_method" : "random", # "random" | "center"
    "batch_size" : 1,
    "bn" : True,
    "random_box": True,
    "random_mask" : True,
    "random_car" : True,
    "random_rotate" : True,
    "other_car_list" : ['per_random_car.txt'], # 'cube_random_car.txt'
    "inpaint_mode" : "non_mask",
    "train_phase" : 1,
    "cont": False,
    "adapt_weight" : "SoftAdapt",
    "update_epoch" : 2,
    "loss_type" : {
        "G": "modify", # "recon" - use only reconstruction loss | "modify" - use full paper loss
        "D": "pconv" # "pix2pix" - use only adversarial loss | "pconv" - use full paper loss
    },
    "loss_lambda":{
        "G":{
            "LAMBDA_adv": 1,    # paper: 1
            "LAMBDA_recon": 100,# paper: 100
            "LAMBDA_perc": 0.05,# paper: 0
            "LAMBDA_tv": 0.1,   # paper: 0
            "LAMBDA_hole": 0,   # paper: 0
            "LAMBDA_valid": 0,  # paper: 0
            "LAMBDA_style": 120 # paper: 0
        },
        "D":{
            "LAMBDA_adv": 1,    # paper: 1
            "LAMBDA_recon": 100,# paper: 100
            "LAMBDA_perc": 0.05,# paper: 0.05
            "LAMBDA_tv": 0.1,   # paper: 0.1
            "LAMBDA_hole": 0,   # paper: 0
            "LAMBDA_valid": 0,  # paper: 0
            "LAMBDA_style": 120 # paper: 120
        }

    },
    "early_stopping" : 5,
    "recon_loss" : {
            "G":"L2", # "L1" | "L2"
            "D": "L2"
        },
    "neck" : 16
    }
    

cfg = Box(config)