from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.utils import down_sample, load_images, orthogonal_init, interpolate_rgb
from src.cnn import MyUNet



# Load ground truths.
ims_gt = load_images()
ims_gt.shape

## Downsample to line scan and interpolate
ims_ds = []
ims_interp = []
interp_mse_list = []
interp_psnr_list = []

train_size = 10
test_size = 15-train_size

print("Downsampling and interpolating...")
for i, im in tqdm(enumerate(ims_gt)):
    # Downsample.
    im_ds, mask = down_sample(im, stride=12, is_torch=False)
    ims_ds.append(im_ds)

    mask2d = (1 - mask[:,:,0]).astype(np.bool)
    # Interpolate.
    im_interp = interpolate_rgb(im_ds, mask2d)
    ims_interp.append(im_interp)
    
    # Calculate MSE and PSNR
    if train_size < i:
        interp_mse_list.append(mse(im_interp, im))
        interp_psnr_list.append(psnr(im_interp, im))
    
ims_ds = np.array(ims_ds)
ims_interp = np.array(ims_interp)

# Convert to torch.
gt_data = torch.tensor(ims_gt.transpose((0,3,1,2)))
ds_data = torch.tensor(ims_ds.transpose((0,3,1,2)))
interp_data = torch.tensor(ims_interp.transpose((0,3,1,2)))

# Split train/test.
Xtrain = interp_data[:train_size]
ytrain = gt_data[:train_size]
Xtest = interp_data[train_size:]
ytest = gt_data[train_size:]

# Load model.
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
print("Running on device", device)
unet = MyUNet()
unet.to(device)

# Train model.
n_epochs = 10
val_size = 1

mse_loss = nn.MSELoss()
optimizer = optim.AdamW(unet.parameters(), lr=0.001)
train_loss = []
test_loss = []

for epoch in range(n_epochs):
    # Training
    unet.train()
    batch_loss = []
    for X, y in zip(Xtrain, ytrain):
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred = unet(X[None, :])
        loss = mse_loss(pred, y[None, :])
        batch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    
    # Get train/validation loss
    if epoch % val_size == 0:
        unet.eval()
        
        train_loss.append(np.mean(batch_loss))
        
        with torch.no_grad():
            batch_loss = []
            for X, y in zip(Xtest, ytest):
                X = X.to(device)
                y = y.to(device)
                pred = unet(X[None, :])
                loss = mse_loss(pred, y[None, :])
                batch_loss.append(loss.item())

            test_loss.append(np.mean(batch_loss))

        print(f"epoch: {epoch} \t train loss: {train_loss[-1]:.6f} \t test loss {test_loss[-1]:.6f}")
    

unet_mse_list = []
unet_psnr_list = []
for i, im in enumrate(ims_interp[train_size:]):
    unet_mse_list.append(mse(im_interp, im))
    unet_psnr_list.append(psnr(im_interp, im))

print('Mean MSE of interpolation:', np.mean(interp_mse_list))
print('Mean PSNR of interpolation:', np.mean(interp_psnr_list))
print('Mean MSE of U-net:', np.mean(unet_mse_list))
print('Mean MSE of U-net:', np.mean(unet_psnr_list))


idx = 14

plt.rcParams.update({'font.size': 22})
fig, axes = plt.subplots(1, 2, figsize=(40, 20))

axes[0].imshow(ims_gt[idx])
axes[0].set_title('Original')
axes[1].imshow(ims_interp[idx])
axes[1].set_title('Interpolated')
plt.savefig('super-duper{}.png'.format(idx))
plt.show()