import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import vgg16
# import sys

from .process_img import preprocess_vgg

def L1_loss_equ(gen_output, target):
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  return l1_loss

def L2_loss_equ(gen_output, target):
  l2_loss = tf.reduce_mean(tf.square(target - gen_output))
  return l2_loss

# contextual encoder loss
def reconstruction_loss(gen_output, target,loss_lambda,recon_loss):
  LAMBDA_recon = loss_lambda["LAMBDA_recon"]
  if(recon_loss == "L1"):
    recon_loss = L1_loss_equ(gen_output, target)
  elif(recon_loss == "L2"):
    recon_loss = L2_loss_equ(gen_output, target)
  else:
    print('Wrong type of reconstruction loss')
    exit()
  total_gen_loss = LAMBDA_recon * recon_loss

  return total_gen_loss, recon_loss


# generator with adversarial loss   
def generator_loss(disc_generated_output, gen_output, target,loss_lambda,recon_loss):
  # L1 Loss
  ## L-hole: l1{elementwise[(1-mask),(out-gt)]}
  ## L-valid: l1{elementwise[(mask),(out-gt)]}
  LAMBDA_recon = loss_lambda["LAMBDA_recon"]

  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output) # adversarial loss

  # Mean absolute error
  # print('target type: ',target)
  # print('gen_output type: ',gen_output)

  if(recon_loss == "L1"):
    recon_loss = L1_loss_equ(gen_output, target)
  elif(recon_loss == "L2"):
    recon_loss = L2_loss_equ(gen_output, target)
  else:
    print('Wrong type of reconstruction loss')
    exit()
  total_gen_loss = gan_loss + (LAMBDA_recon * recon_loss)

  return total_gen_loss, gan_loss, recon_loss

# perceptual loss ----------------------------------
class LossNetwork(tf.keras.models.Model):
    def __init__(self, style_layers = ['block1_conv2',
                                       'block2_conv2',
                                       'block3_conv3',
                                       'block4_conv3']):
        super(LossNetwork, self).__init__()
        vgg = vgg16.VGG16(include_top=False, weights='imagenet')
        vgg.trainable = False
        model_outputs = [vgg.get_layer(name).output for name in style_layers]
        self.model = tf.keras.models.Model(vgg.input, model_outputs)
        # mixed precision float32 output
        self.linear = layers.Activation('linear', dtype='float32')

    def call(self, x):
        x = x * 255.0
        x = vgg16.preprocess_input(x)
        x = self.model(x)
        return self.linear(x)
    
def perceptual_loss(gen_img,real_img):
  loss_network = LossNetwork()
  target_feature_maps = loss_network(real_img) # output 4 layer from model (input 8)
  # print(len(target_feature_maps))
  output_feature_maps = loss_network(gen_img)
  per_loss = tf.add_n([tf.reduce_mean(tf.abs(target_feature_maps[2]-output_feature_maps[2]))]) #L1
  # per_loss = tf.add_n(tf.reduce_mean((target_feature_maps[2]-output_feature_maps[2])**2)) #MSE
  # return tf.reduce_mean(tf.square(target_feature_maps[2] - output_feature_maps[2]), axis=[1]) #MSE
  return per_loss
# --------------------------------------------------------------------------------------------------------
# style loss ------------------------------------------------------------
def vgg_layers(layer_names):
  """ Creates a VGG model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on ImageNet data
  vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
  vgg.trainable = False

  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name: value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}

def style_content_loss(gen_img,real_img,recon_loss):
    # variable -------------------
    # content_weight=1e4

    content_layers = ['block1_conv2',
                      'block2_conv2',
                      'block3_conv3',
                      'block4_conv3']

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1']

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)
    # -------------------------------------------
    extractor = StyleContentModel(style_layers, content_layers)
    outputs = extractor(gen_img)
    style_targets = extractor(real_img)['style']
    content_targets = extractor(real_img)['content']


    style_outputs = outputs['style']
    content_outputs = outputs['content']
    if(recon_loss == "L1"):
      style_loss = tf.add_n([tf.reduce_mean(tf.abs(style_outputs[name]-style_targets[name])) 
                            for name in style_outputs.keys()]) #L1
      content_loss = tf.add_n([tf.reduce_mean(tf.abs(content_outputs[name]-content_targets[name])) 
                            for name in content_outputs.keys()]) #L1
    elif(recon_loss == "L2"):
      style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()]) #L2
      content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                           for name in content_outputs.keys()]) #L2
      
    style_loss = style_loss / num_style_layers
    content_loss = content_loss / num_content_layers
    # loss = style_loss

    # content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
    #                          for name in content_outputs.keys()])
    # content_loss *= content_weight / num_content_layers
    # loss = style_loss + content_loss
    return style_loss, content_loss

# -----------------------------------------------------------------------------------------------------
# ---------------------total variation loss -----------------------------
def high_pass_x_y(image):
  x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
  y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

  return x_var, y_var

def total_variation_loss(image):
  x_deltas, y_deltas = high_pass_x_y(image)
  return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))
# --------------------------------------------------------------------------


def discriminator_loss(disc_real_output, disc_generated_output, predict_img, inpaint_img,mask, target, loss_type = 'pconv', loss_lambda={}, recon_loss="L1"):
  LAMBDA_adv = loss_lambda["LAMBDA_adv"]
  LAMBDA_perc = loss_lambda["LAMBDA_perc"]
  LAMBDA_tv = loss_lambda["LAMBDA_tv"]
  LAMBDA_hole = loss_lambda["LAMBDA_hole"]
  LAMBDA_valid = loss_lambda["LAMBDA_valid"]
  LAMBDA_style =loss_lambda["LAMBDA_style"]

  # if 'perc' in exclude_loss:
  #    LAMBDA_perc = 0
  # if 'tv' in exclude_loss:
  #    LAMBDA_tv = 0
  # if 'style' in exclude_loss:
  #    LAMBDA_style = 0
  # if 'valid' in exclude_loss:
  #    LAMBDA_valid = 0
  # if 'hole' in exclude_loss:
  #    LAMBDA_hole = 0

  # perceptual
  ## p(inpaint) + p(out)

  #style
  ## s(inpaint) + s(out)

  # LAMBDA = 100
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output) # adversarial loss

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output) # adversarial loss
  adv_loss = real_loss + generated_loss


  # perceptual loss -----------------------------------------
  # tf.convert_to_tensor(inpaint_img, dtype=tf.float32)
  # gen_img = tf.convert_to_tensor([tf.image.resize(x, [224,224], method='bilinear') for x in inpaint_img], dtype=tf.float32)
  # real_img = tf.convert_to_tensor([tf.image.resize(x, [224,224], method='bilinear') for x in target], dtype=tf.float32)
  out_img = preprocess_vgg(predict_img)
  comp_img = preprocess_vgg(inpaint_img)
  real_img = preprocess_vgg(target)


  # perc_loss_out = perceptual_loss(out_img,real_img)
  # perc_loss_comp = perceptual_loss(comp_img,real_img)
  # perc_loss = perc_loss_comp + perc_loss_out
  # print(perc_loss)
  # perc_loss = 1e-6*(sum(sum(sum(perc_loss))))
  # perc_loss= 0
  # ---------------------------------------------------------
  # style_content loss --------------------------------
  style_loss_out, perc_loss_out = style_content_loss(out_img,real_img,recon_loss)
  style_loss_comp, perc_loss_comp = style_content_loss(comp_img,real_img,recon_loss)
  style_loss =  style_loss_out + style_loss_comp
  perc_loss = perc_loss_comp + perc_loss_out
  # print(style_loss)
  # ----------------------------------------------------
  # total variation loss-------------------------------
  tv_loss =  total_variation_loss(inpaint_img)
  #---------------------------------------------------
  # L1 loss --------------------------------------------
  if(recon_loss == "L1"):
    recon_hole = L1_loss_equ(((1 - mask) * predict_img),((1 - mask) * target))
    recon_valid = L1_loss_equ((mask * predict_img),(mask * target))
  elif(recon_loss == "L2"):
    recon_hole = L2_loss_equ(((1 - mask) * predict_img),((1 - mask) * target))
    recon_valid = L2_loss_equ((mask * predict_img),(mask * target))
  # l1_loss_hole = tf.reduce_mean(tf.abs(((1 - mask) * predict_img) - ((1 - mask) * target)))
  # l1_loss_valid = tf.reduce_mean(tf.abs((mask * predict_img) - (mask * target)))

  loss_hole = recon_hole
  loss_valid =  recon_valid
  # ------------------------------------------------------------------
  if(loss_type == 'pconv'):
    total_disc_loss = (LAMBDA_adv * adv_loss) + (LAMBDA_perc * perc_loss) + (LAMBDA_style * style_loss) + (LAMBDA_tv * tv_loss) + (LAMBDA_hole * loss_hole) + (LAMBDA_valid * loss_valid)

    return total_disc_loss,adv_loss, perc_loss, style_loss, tv_loss, loss_hole, loss_valid
  elif(loss_type == 'p2p'):
    total_disc_loss = (LAMBDA_adv * adv_loss)
    return total_disc_loss
  else:
     return "wrong type"
# ---------------------------------------------------------

# Modified Generator loss (for L1)------------------------------------------------------------
def modified_generator_loss(predict_img, inpaint_img,mask, target, loss_type = 'pconv', loss_lambda={}, recon_loss="L1"):
  LAMBDA_recon = loss_lambda["LAMBDA_recon"]
  LAMBDA_perc = loss_lambda["LAMBDA_perc"]
  LAMBDA_tv = loss_lambda["LAMBDA_tv"]
  LAMBDA_hole = loss_lambda["LAMBDA_hole"]
  LAMBDA_valid = loss_lambda["LAMBDA_valid"]
  LAMBDA_style =loss_lambda["LAMBDA_style"]

  # perceptual loss -----------------------------------------
  # tf.convert_to_tensor(inpaint_img, dtype=tf.float32)
  # gen_img = tf.convert_to_tensor([tf.image.resize(x, [224,224], method='bilinear') for x in inpaint_img], dtype=tf.float32)
  # real_img = tf.convert_to_tensor([tf.image.resize(x, [224,224], method='bilinear') for x in target], dtype=tf.float32)
  out_img = preprocess_vgg(predict_img)
  comp_img = preprocess_vgg(inpaint_img)
  real_img = preprocess_vgg(target)


  # perc_loss_out = perceptual_loss(out_img,real_img)
  # perc_loss_comp = perceptual_loss(comp_img,real_img)
  # perc_loss = perc_loss_comp + perc_loss_out
  # print(perc_loss)
  # perc_loss = 1e-6*(sum(sum(sum(perc_loss))))
  # perc_loss= 0
  # ---------------------------------------------------------
  # style_content loss --------------------------------
  style_loss_out, perc_loss_out = style_content_loss(out_img,real_img,recon_loss)
  style_loss_comp, perc_loss_comp = style_content_loss(comp_img,real_img,recon_loss)
  style_loss =  style_loss_out + style_loss_comp
  perc_loss = perc_loss_comp + perc_loss_out
  # print(style_loss)
  # ----------------------------------------------------
  # total variation loss-------------------------------
  tv_loss =  total_variation_loss(inpaint_img)
  #---------------------------------------------------
  # L1 loss --------------------------------------------
  if(recon_loss == "L1"):
    recon_hole = L1_loss_equ(((1 - mask) * predict_img),((1 - mask) * target))
    recon_valid = L1_loss_equ((mask * predict_img),(mask * target))
    recon_whole = L1_loss_equ(predict_img,target)
  elif(recon_loss == "L2"):
    recon_hole = L2_loss_equ(((1 - mask) * predict_img),((1 - mask) * target))
    recon_valid = L2_loss_equ((mask * predict_img),(mask * target))
    recon_whole = L2_loss_equ(predict_img,target)
  # l1_loss_hole = tf.reduce_mean(tf.abs(((1 - mask) * predict_img) - ((1 - mask) * target)))
  # l1_loss_valid = tf.reduce_mean(tf.abs((mask * predict_img) - (mask * target)))

  loss_hole = recon_hole
  loss_valid =  recon_valid
  # ------------------------------------------------------------------
  if(loss_type == "recon"):
    total_gen_loss = (LAMBDA_recon * recon_whole)
    return total_gen_loss, recon_whole
  elif(loss_type == "modify"):
    total_gen_loss = (LAMBDA_recon * recon_whole)+(LAMBDA_perc * perc_loss) + (LAMBDA_style * style_loss) + (LAMBDA_tv * tv_loss) + (LAMBDA_hole * loss_hole) + (LAMBDA_valid * loss_valid)
    return total_gen_loss,recon_whole, perc_loss, style_loss, tv_loss, loss_hole, loss_valid
  else:
    return ("wrong loss type")