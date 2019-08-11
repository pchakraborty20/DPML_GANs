import os
import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.autograd.variable import Variable
import torchvision
from torchvision import transforms, datasets

from utils import Logger



module = torch.nn.Module

#Potentially helpful methods? [vestigial]
def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

#Define Discriminator (takes images from GAN and compares to original z_vec image)
class DiscriminatorNet(torch.nn.Module):

    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features= 256
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.LeakyReLU(.2),
            nn.Dropout(.1)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(.2),
            nn.Dropout(.1)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(.2),
            nn.Dropout(.1)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(32, n_out),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out()
        return x

##-----------------------------------------------
##Define Generator (takes random noise vector as input, outputs image)
##-----------------------------------------------
class GeneratorNet(torch.nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 128
        n_out = 256

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(.2),
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(.2),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(.2),
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

generator = GeneratorNet()

def noise(size):
    n = Variable(torch.randn(size, 100))
    return n

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

loss = nn.BCELoss()

#Disc/Gen Targets => it's trying to always identify reals as 1s
def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data


def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, ones_target(N) )
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()

    optimizer.step()

    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer, fake_data):
    N = fake_data.size(0)
    optimizer.zero_grad()

    prediction = discriminator(fake_data)

    error = loss(prediction, ones_target(N))
    error.backward()

    optimizer.step()

    return error

#Setup Training Data
amt_data = 1000'''?'''
data = [torch.new_empty((1,128), requires_grad = True'''?''') for i in range(amt_data)]
numBatches = len(data)
test_noise = noise(num_test_samples)


###################################Rewrite for this actual problem############################################
# Create logger instance
logger = Logger(model_name='VGAN', data_name='MNIST')
# Total number of epochs to train
num_epochs = 200
for epoch in range(num_epochs):
    for n_batch, (real_batch,_) in enumerate(data_loader):
        N = real_batch.size(0)
        # 1. Train Discriminator
        real_data = Variable(images_to_vectors(real_batch))
        # Generate fake data and detach 
        # (so gradients are not calculated for generator)
        fake_data = generator(noise(N)).detach()
        # Train D
        d_error, d_pred_real, d_pred_fake = \
              train_discriminator(d_optimizer, real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(N))
        # Train G
        g_error = train_generator(g_optimizer, fake_data)
        # Log batch error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)
        # Display Progress every few batches
        if (n_batch) % 100 == 0: 
            test_images = vectors_to_images(generator(test_noise))
            test_images = test_images.data
            logger.log_images(
                test_images, num_test_samples, 
                epoch, n_batch, num_batches
            );
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, d_pred_real, d_pred_fake
            )




        
