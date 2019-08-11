#Simple random Z_vec + noise -> gradDesc -> original

import torch
from torch import optim, nn
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample, convert_to_images,save_as_images)
from PIL import Image
import random
import os

model = BigGAN.from_pretrained('biggan-deep-256')

truncation = .4
batches = 1
maxNoise = .4

def mse_loss(first, target):
    return torch.sum((first - target) ** 2)

    #Save Pictures
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def writeTrial(trial, runs, filename):
    pass

trials = int(input('How many sets?\n'))
##record = str(input('enter recording file name:  '))
##ensure_dir(record)
def recover_latent(trial):
    trueZ = truncated_noise_sample(truncation=truncation, batch_size=batches)
    noise = truncated_noise_sample(truncation = maxNoise, batch_size = batches)
    class_vec = one_hot_from_names(['fountain'], batch_size=batches)

    z = torch.from_numpy(trueZ + noise)
    print(z)
    ##print('diff:\n', z-trueZ)
    opt = optim.Adam([z.requires_grad_()])
    with torch.no_grad():
        trueZImg = model(torch.from_numpy(trueZ), torch.from_numpy(class_vec), truncation).requires_grad_()
    zImg = model(z, torch.from_numpy(class_vec), truncation).requires_grad_()

    zImg0 = zImg.clone()

    i = 0
    while(i < 5):
        lf = nn.MSELoss()
        loss = lf(zImg, trueZImg)
        loss.backward()
        opt.step()
        opt.zero_grad()
        i += 1
        print(i, ': ImageMSE: ', mse_loss(zImg, trueZImg), '\sVecMSE: ', mse_loss(z, torch.from_numpy(trueZ)))
        zImg = model(z, torch.from_numpy(class_vec) , truncation).requires_grad_()
        
    ##with torch.no_grad():
    ##zImg = model(torch.from_numpy(z, class_vec, truncation)

    trial = 1

    #Save Images
    saveOriginal = 'output/' + str(trial) + '_original'
    saveNoisy = 'output/' + str(trial) + '_noisy'
    saveFixed = 'output/' + str(trial) + '_fixed'
    ensure_dir(saveOriginal)
    save_as_images(trueZImg, saveOriginal)
    ensure_dir(saveNoisy)
    save_as_images(zImg0, saveNoisy)
    ensure_dir(saveFixed)
    save_as_images(zImg, saveFixed)

    #Save vectors
    saveOriginal = 'output/' + str(trial) + 'originalVec_.pt'
    saveNoisy = 'output/' + str(trial) + '_noisyVec.pt'
    saveFixed = 'output/' + str(trial) + '_fixedVec.pt'
    ensure_dir(saveOriginal)
    torch.save(trueZImg, saveOriginal)
    ensure_dir(saveNoisy)
    torch.save(zImg0, saveNoisy)
    ensure_dir(saveFixed)
    torch.save(zImg, saveFixed)

    #Trial Metadata

for i in range(trials):
    main(i)

##saveZ = 'output/zRecovered_' + str(trial)
##saveTrueZ = 'output/zTrue_' + str(trial)
##ensure_dir(
##torch.save
