import os
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from dip import EncDec
from utils import imread
from utils import gaussian
from scipy import ndimage
from scipy import signal
import cv2
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Load clean and noisy image
# im = imread('../data/denoising/saturn.png')
# noise1 = imread('../data/denoising/saturn-noisy.png')
im = imread('../data/denoising/lena.png')
noise1 = imread('../data/denoising/lena-noisy.png')

error1 = ((im - noise1)**2).sum()

print('Noisy image SE: {:.2f}'.format(error1))

plt.figure(1)

# Original image
plt.subplot(141)
plt.imshow(im, cmap='gray')
plt.title('Input')

# Noisy version
plt.subplot(142)
plt.imshow(noise1, cmap='gray')
plt.title('Noisy image SE {:.2f}'.format(error1))

# Apply Gaussian filter
plt.subplot(143)
gaussian_filter = gaussian(7, 2)
im_gauss = ndimage.convolve(noise1, gaussian_filter, mode="nearest")
error_gauss = ((im - im_gauss)**2).sum()
plt.imshow(im_gauss, cmap='gray')
plt.title('Gaussian SE {:.2f}'.format(error_gauss))

# Apply Median filter
plt.subplot(144)
im_med = signal.medfilt(noise1, 7)
error_med = ((im - im_med)**2).sum()
plt.imshow(im_med, cmap='gray')
plt.title('Median SE {:.2f}'.format(error_med))
plt.show(block=True)


################################################################################
# Denoising algorithm (Deep Image Prior)
################################################################################

# Create network
model = EncDec()

# Loads noisy image and sets it to the appropriate shape
noisy_img = torch.FloatTensor(noise1).unsqueeze(0).unsqueeze(0).transpose(2, 3)
clean_img = torch.FloatTensor(im).unsqueeze(0).unsqueeze(0).transpose(2,3)
# Creates \eta (noisy input)
eta = torch.randn(*noisy_img.size())
eta.detach()


opt = optim.Adam(model.parameters(), lr = 0.01)
n_epochs = 1000
mse = nn.L1Loss()
train_error = []
test_error = []

# Define an array to store the average image after every 100 epochs
average = np.linspace(200, 300, 100)
avg_img = np.zeros((256, 384))

# Training loop
for i in tqdm(range(n_epochs)):
    opt.zero_grad()

    # Forward pass through the network
    out_eta = model(eta)

    # Compute the average image at 100-epoch intervals
    if i in average:
        avg_img += out_eta.clone()[0, 0, :, :].transpose(0,1).detach().numpy()

    # Compute the training loss
    loss_train = mse(out_eta, noisy_img)
    train_error.append(np.log(loss_train.item()))

    # Compute the test loss
    loss_test = mean_squared_error(out_eta.clone().detach().cpu().numpy()[0, 0, :, :], clean_img.detach().cpu().numpy()[0, 0, :, :])
    test_error.append(np.log(loss_test))

    # Backpropagation and optimization step
    loss_train.backward()
    opt.step()

x = np.array(list(range(n_epochs)))
plt.plot(x, np.array(train_error), color='blue', label = "train_error", linewidth=1.5)
plt.plot(x, np.array(test_error), color='green', label = "test_error", linewidth=1.5)
plt.xlabel("No. of Epochs", fontsize=14)
plt.ylabel("MSE Loss", fontsize=14)
plt.legend(fontsize=12)
plt.show()


out = model(eta)
out_img = out[0, 0, :, :].transpose(0,1).detach().numpy()

cv2.imwrite("/Users/Christina/Documents/dip.png", out_img)

error1 = ((im - noise1)**2).sum()
error2 = ((im - out_img)**2).sum()

plt.figure(3)
plt.axis('off')

plt.subplot(131)
plt.imshow(im, cmap='gray')
plt.title('Input')

plt.subplot(132)
plt.imshow(noise1, cmap='gray')
plt.title('SE {:.2f}'.format(error1))

plt.subplot(133)
plt.imshow(out_img, cmap='gray')
plt.title('SE {:.2f}'.format(error2))

plt.show()
