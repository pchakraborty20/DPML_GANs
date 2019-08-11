import torch
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)
from PIL import Image

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
model = BigGAN.from_pretrained('biggan-deep-256')

truncation = .4

# Prepare a input
nv = torch.load('Experiments/nv.pt')
class_vector = torch.load('Experiments/cv.pt')

##nv2 = nv.sin()
##nv3 = nv2.sin()
##nv4 = nv3.sin()
##nv5 = nv4.sin()

##nv2 = nv.cos().abs()-.5
##nv3 = nv2.cos().abs()-.5
##nv4 = nv3.cos().abs()-.5
##nv5 = nv4.cos().abs()-.5

##nv2 = nv.cos()
##nv3 = nv2.cos()
##nv4 = nv3.cos()
##nv5 = nv4.cos()

##nv2 = nv.tan()
##nv3 = nv2.tan()
##nv4 = nv3.tan()
##nv5 = nv4.tan()





##nv4 = nv.abs()/nv
##nv5 = nv4*-1

noise_vector = torch.cat((nv, nv2, nv3, nv4, nv5), 0)
class_vector = torch.cat((class_vector, class_vector, class_vector, class_vector, class_vector), 0)


# Generate an image
with torch.no_grad():
    output = model(noise_vector, class_vector, truncation)


# If you have a sixtel compatible terminal you can display the images in the terminal
# (see https://github.com/saitoha/libsixel for details)
#display_in_terminal(output)

#Save results as png images
save_as_images(output, 'output/Static/static_0')
