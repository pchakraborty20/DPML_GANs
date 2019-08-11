import torch
    from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                           save_as_images, display_in_terminal, convert_to_images)
    from PIL import Image
import random
import os

import imageio

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
model = BigGAN.from_pretrained('biggan-deep-256')


# Prepare a input
batches = 1
truncation = 1
classnam = str(input('Enter desired class:\n'))
numpics = int(input('Enter desired number of pictures:\n'))
large_batches = True if str(input('Large Batches?\n')) == "y" else False
jitter = True if str(input('Jitter?\n')) == "y" else False
if jitter:
    super_jitter = True if str(input('SuperJitter?\n')) == "y" else False
    if super_jitter:
        minJ = float(input('MinJitter:\n'))

class_vector = one_hot_from_names([classnam], batch_size=batches)
noise_vector = truncated_noise_sample(truncation=truncation, batch_size=batches)

### All in tensors
nv = torch.from_numpy(noise_vector)
class_vector = torch.from_numpy(class_vector)
##
##torch.save(nv, 'Experiments/nv.pt')
##torch.save(class_vector, 'Experiments/cv.pt')

##class_vector = torch.load('Experiments/cv.pt')

#0?
##nv2 = nv.cos().abs()-.5
##nv3 = nv2.cos().abs()-.5
##nv4 = nv3.cos().abs()-.5
##nv5 = nv4.cos().abs()-.5

#0?
##nv2 = nv.cos()
##nv3 = nv2.cos()
##nv4 = nv3.cos()
##nv5 = nv4.cos()

#x = 1
##nv2 = nv.sin()
##nv3 = nv2.sin()
##nv4 = nv3.sin()
##nv5 = nv4.sin()

#x = 2

#x = 3
##nv2 = (nv.abs().log() + min(nv.abs().log()))/max(nv.abs().log())
##nv3 = (nv2.abs().log() + min(nv2.abs().log()))/max(nv2.abs().log())
##nv4 = (nv3.abs().log() + min(nv3.abs().log()))/max(nv3.abs().log())
##nv5 = (nv4.abs().log() + min(nv4.abs().log()))/max(nv4.abs().log())
##
##
####nv4 = nv.abs()/nv
##nv5 = nv4*-1

#x = 4
##nv2 = torch.pow(nv, 3)
##nv3 = torch.pow(nv2, 3)
##nv4 = torch.pow(nv3, 3)
##nv5 = torch.pow(nv4, 3)

#x = 6
##nv2 = nv.sin().sin().sin().sin().sin().sin()
##nv3 = nv2.sin().sin().sin().sin().sin().sin()
##nv4 = nv3.sin().sin().sin().sin().sin().sin()
##nv5 = nv4.sin().sin().sin().sin().sin().sin()

#x = 7
##nv2 = torch.pow(nv, 3)
##nv3 = nv2.sin().sin().sin().sin().sin().sin()
##nv4 = torch.pow(nv3, 3).asin().asin().asin().asin().asin().asin().asin()
##nv5 = nv4.cos().abs()-.5

#x = 8
##nv2 = torch.pow(nv, 3)
##nv3 = nv2.asin().asin().asin().asin().asin().asin().asin().asin()
##nv4 = torch.pow(nv3, .333)
##nv5 = nv4.asin().asin().asin().asin().asin().asin().asin().asin()

#x = 9
##nv2 = nv.abs()/2
##nv3 = nv2.abs()/2
##nv4 = nv3.abs()/2
##nv5 = nv4.abs()/2

#x = 10
##nv2 = nv.abs()-.5
##nv3 = nv2.abs()-.25
##nv4 = nv3.abs()-.125
##nv5 = nv4.abs()-.0625

#x = 11
##nv2 = nv.abs()-.5
##nv3 = nv2.abs()+.25
##nv4 = nv3.abs()-.125
##nv5 = nv4.abs()+.0625

#x = 12 -> Jitter Extremes
#create jitter tensor
nvlist = [nv]
##for i in range(4):
##    jitter = torch.tensor([(random.random()/2-.5) for i in range(len(nv[0].tolist()))])
##    nv2 = nvlist[i].clone()
##    temp = nvlist[i].clone().tolist()
##    nv2numpy = nv2.tolist()
##    for j in range(len(nv2numpy[0])):
##        if nv2numpy[0][j] > .5:
##            nv2numpy[0][j] = temp[0][j] + jitter[j]
##        nv2 = torch.tensor(nv2numpy)
####  nv2[nv.abs() > .5] = .75
##    nv2 += jitter
##    nv2numpy = nv2.tolist()
##    jitternumpy = jitter.tolist()
##    #, ' ', nv2.size(), ' ', temp.size())
####    nv2[nv2[0] - jitter == temp] = temp
##    for j in range(len(nv2numpy[0])):
##        if nv2numpy[0][j] - jitternumpy[j] == temp[0][j]:
##            nv2numpy[0][j] = temp[0][j]
##        nv2 = torch.tensor(nv2numpy)
##    nvlist.append(nv2)


#**thresh is upper bound for JitterInner, and lower bound for standard Jitter
def jitterExt(tensor, thresh):
    jitter = torch.tensor([(random.random()*thresh-thresh) for i in range(len(nv[0].tolist()))])
    ntens = tensor.clone()
    chk = ntens[0]
    for i in range(len(chk)):
        if abs(chk[i]) > thresh:
            print(ntens[0][i])
            #Jitter
            ntens[0][i] = ((thresh+1)/2)*abs(chk[i])/chk[i] + jitter[i]
            print(ntens[0][i])
    print(ntens-tensor)
    return ntens

def superJitter(tensor, thresh, minJ):
    ntens = jitterExt(tensor, thresh)
    chk = ntens[0]
    for i in range(len(chk)):
        diff = chk[i]-tensor[0][i]
        if abs(chk[i]) > thresh and abs(diff) < minJ:
            ntens[0][i] -= .05*abs(chk[i])/chk[i]
    print(ntens-tensor)
    return ntens



##x = 12
##threshold = .2

##x = 13
##threshold = .5

x = 14
threshold = .35

##x = 15
##threshold = .7

nvlist = [nv]

for i in range(numpics-1):
    if not super_jitter:
        nvlist.append(jitterExt(nvlist[0], threshold))
    else:
        nvlist.append(superJitter(nvlist[i], threshold, minJ))

##print(nvlist)


noise_vector = torch.cat(nvlist, 0)

class_vectorX = [class_vector]
for i in range(numpics-1):
    class_vectorX.append(class_vector)
class_vector = torch.cat((class_vectorX), 0)
##class_vector = torch.tensor(class_vectorX)


# Generate an image
with torch.no_grad():
    output = model(noise_vector, class_vector, truncation)


# If you have a sixtel compatible terminal you can display the images in the terminal
# (see https://github.com/saitoha/libsixel for details)
#display_in_terminal(output)

# Save results as png images
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

saveFile = str('output/RepJitterExp/')
if large_batches:
    saveFile += str('_LargeBatches/')

saveFile += str(str(classnam.capitalize()) + '/')
if super_jitter:
    saveFile += str('SuperJitter' + str(minJ) + '/')
saveFile += str(classnam + '_' + str(x))


ensure_dir(saveFile)

save_as_images(output, saveFile)


#Save pngs as a .gif

imgs = convert_to_images(output)

saveGIF = saveFile + '/anim.gif'

ensure_dir(saveGIF)

imgs[0].save(saveGIF,
               save_all=True,
               append_images=imgs[1:],
               duration=100,
               loop=0)





'''noise_vector = noise_vector.sin_()

with torch.no_grad():
    output = model(noise_vector, class_vector, truncation)

save_as_images(output)'''

