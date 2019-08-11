import torch
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)
import numpy as np
import json
try:
    import Image
except ImportError:
    from PIL import Image

# Load pre-trained model
model = BigGAN.from_pretrained('biggan-deep-256')
truncation = 0.4

# List of classes to interpolate between
classes = ['soup bowl', 'wolf']
class_vector = torch.from_numpy(one_hot_from_names(classes, batch_size=len(classes)))

# Defining a nonlinear spacing function to ease in/out of categories
def spacing_func(y, branch):
	return (1+branch*y*np.sqrt((2-np.power(y,2)).clip(0)))/2

p_range = np.concatenate((spacing_func(np.linspace(1,np.sqrt(2),10,endpoint=False), -1),
						  spacing_func(np.linspace(np.sqrt(2),1,10,endpoint=False), 1)))

# Creating mixed category vectors
nIm = len(classes)*len(p_range)
interp_class = torch.Tensor(nIm, class_vector.shape[1])
i = 0
for interp in range(len(classes)):
	for p in p_range:
		interp_class[i] = np.sqrt(1-p)*class_vector[interp,:]+np.sqrt(p)*class_vector[(interp+1)%len(classes),:]
		i += 1

# Sampling a single latent noise vector
noise_vector = torch.from_numpy(np.repeat(truncated_noise_sample(truncation=truncation, seed=0), nIm, axis=0)).sin_()

# Generate images
with torch.no_grad():
    output = model(noise_vector, interp_class, truncation)

# Save results as png images
save_as_images(output, file_name='output/out2')
