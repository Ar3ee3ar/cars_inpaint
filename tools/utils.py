import wandb
from dotenv import load_dotenv
import os
from os.path import join, dirname
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def generate_images(model, test_input,mask, tar,model_name, log=True):
  if(model_name == "pconv"):
    gen_output = model([test_input,mask], training=True)
    inpaint_img = [mask * tar[0] + (1 - mask) * gen_output[0]]
    display_list = [test_input[0], tar[0], gen_output[0],inpaint_img[0][0]]
  elif(model_name == "p2p"):
    gen_output = model(test_input, training=True)
    gen_output_norm = (gen_output[0] * 0.5) + 0.5
    test_input_norm = (test_input[0] * 0.5) + 0.5
    tar_norm = (tar[0] * 0.5) + 0.5
    inpaint_img = [mask * tar_norm + (1 - mask) * gen_output_norm]
    display_list = [test_input_norm, tar_norm, gen_output_norm,inpaint_img[0][0]]

#   plt.figure(figsize=(15, 15))

  # gen_output = prediction.numpy()

  # get_masked_image = gen_output[0].copy() # get image in mark area
  # get_masked_image[mask[0]>0.] = 1 # bitwise
  # # get_masked_image[mask[0]==0] = 1 # original
  # r,c,ch = np.where(get_masked_image==1)
  # get_masked_image[(r,c)] = test_input[0][(r,c)]
  # inpaint_img = [get_masked_image]

  # gen_output = generator([input_image,mask], training=True)
  # replace img
  # tf.identity(gen_output[0])
  # mask_area = tf.where(tf.equal(mask[0], 1.), mask[0],tf.identity(gen_output[0]))
  # # print(mask_area)
  # inpaint_img = [tf.where(tf.equal(mask_area, 1.),test_input[0], mask_area)]
  # original
  # mask_area = tf.where(tf.equal(mask[i], 0), mask[i], tf.identity(gen_output[i]))
  # inpaint = tf.where(tf.equal(mask_area, 0),input_image[i], mask_area)

  
  title = ['Input Image', 'Ground Truth', 'Predicted Image', 'Inpaint Image']

  for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i])
    plt.axis('off')
  # plt.show()
  plt.savefig('test_save.png')
  if(log):
    wandb.log({"masked_images": [wandb.Image(display_list[0])]})
    wandb.log({"predictions": [wandb.Image(display_list[2])]})
    wandb.log({"inpaint": [wandb.Image(display_list[3])]})
    wandb.log({"labels": [wandb.Image(display_list[1])]})

def wandb_log():
  dotenv_path = join(dirname(__file__), '.env')
  load_dotenv(dotenv_path)
  wandb.login(key=os.environ.get('WANDB_KEYS'))
  # initial wandb log
  wandb.tensorboard.patch(root_logdir="")
  wandb.init(entity='arzeezar-l', project="img_inpaint")
  return wandb.run.dir

@tf.function
def view_test(testgen,generator,model_name,batch_size):
  # new perceptual
  ## Legend: Original Image | Mask generated | Inpainted Image | Ground Truth

  ## Examples
  if(batch_size == 1):
    rows = batch_size + 1
  else:
    rows = batch_size
  print(rows)
  sample_idx = 25
  [masked_images, masks], sample_labels = testgen[sample_idx]

  fig, axs = plt.subplots(nrows=rows, ncols=4, gridspec_kw = {'wspace':0, 'hspace':0}, figsize=(15,15))
  if(model_name == "pconv"):
      inputs = [masked_images, masks]
      impainted_image = generator.predict(inputs)
  elif(model_name == "p2p"):
    masked_images = (masked_images - 0.5)/0.5 # normalize [-1,1]
    inputs = masked_images
    impainted_image = generator(inputs, training=True)
    impainted_image = (impainted_image* 0.5) + 0.5 # de-normalize [0,1]
    masked_images = (masked_images* 0.5) + 0.5 # de-normalize [0,1]

  # print(impainted_image.shape)

  for i in range(rows):
    axs[i][0].imshow(masked_images[i])
    # axs[i][1].imshow(masks[i])
    # get_masked_image = impainted_image[0].copy()
    # get_masked_image[masks[i]>0] = 1
    # r,c,ch = np.where(get_masked_image==1)
    # get_masked_image[(r,c)] = masked_images[i][(r,c)]
    get_masked_image = masks[i] * sample_labels[i] + (1 - masks[i]) * impainted_image[i]
    axs[i][1].imshow(impainted_image[i])
    axs[i][2].imshow(get_masked_image)
    axs[i][3].imshow(sample_labels[i])

  for ax in fig.axes:
    ax.axison = False

  plt.show()



  # python training.py --dir "D:/inpaint_gan/" --wandb True --weightG "D:/inpaint_gan/weight/1999G_weight.h5" --weightD "D:\inpaint_gan\weight\1999D_weight.h5" --save 1 --step 2