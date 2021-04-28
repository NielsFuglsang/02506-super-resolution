import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

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
interp_ssim_list = []

train_size = 10
test_size = 15-train_size

print("Downsampling and interpolating...")
for i, im in enumerate(ims_gt):
    # Downsample.
    im_ds, mask = down_sample(im, stride=12, is_torch=False)
    ims_ds.append(im_ds)

    mask2d = (1 - mask[:,:,0]).astype(bool)
    # Interpolate.
    im_interp = interpolate_rgb(im_ds, mask2d)
    ims_interp.append(im_interp)
    
    # Calculate MSE and PSNR
    if train_size < i:
        interp_mse_list.append(mse(im_interp, im))
        interp_psnr_list.append(psnr(im_interp, im))
        interp_ssim_list.append(ssim(im_interp, im, multichannel=True))
    
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
n_epochs = 100
val_size = 5

mse_loss = nn.MSELoss()
optimizer = optim.AdamW(unet.parameters(), lr=0.001)
train_losses = []
test_losses = []

for epoch in range(n_epochs):
    # Training
    unet.train()
    for X, y in zip(Xtrain, ytrain):
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred = unet(X[None, :])
        loss = mse_loss(pred, y[None, :])
        loss.backward()
        optimizer.step()
    
    # Get train/validation loss
    if epoch % val_size == 0:
        unet.eval()
        
        with torch.no_grad():
            train_loss = []
            for X, y in zip(Xtrain, ytrain):
                X = X.to(device)
                y = y.to(device)
                pred = unet(X[None, :])
                loss = mse_loss(pred, y[None, :])
                train_loss.append(loss.item())
            train_losses.append(np.mean(train_loss))

            test_loss = []
            for X, y in zip(Xtest, ytest):
                X = X.to(device)
                y = y.to(device)
                pred = unet(X[None, :])
                loss = mse_loss(pred, y[None, :])
                test_loss.append(loss.item())
            test_losses.append(np.mean(test_loss))


        print(f"epoch: {epoch} \t train loss: {train_losses[-1]:.6f} \t test loss {test_losses[-1]:.6f}")


fig, ax = plt.subplots(1, 1, figsize=(10,5))
ax.plot(np.arange(0, n_epochs, val_size), train_losses, linewidth=3, label='train error')
ax.plot(np.arange(0, n_epochs, val_size), test_losses, linewidth=3, label='test error')
ax.set_yscale('log')
ax.set_ylabel('MSE')
ax.legend()
ax.set_title('Convergence')
plt.savefig('figures/convergence.png', bbox_inches='tight')
plt.show()


# Get reconstructed data in numpy.
unet.eval()
ims_superres = np.zeros(ims_gt.shape)
for i in range(15):
    with torch.no_grad():
        X = interp_data[None, i].to(device)
        pred = unet(X)
        pred_np = pred[0].cpu().detach().numpy().transpose((1,2,0))
        pred_np = np.clip(pred_np, 0, 1)
        ims_superres[i] = pred_np

# Get MSE and PSNR 
unet_mse_list = []
unet_psnr_list = []
unet_ssim_list = []

for i in range(train_size, 15):
    unet_mse_list.append(mse(ims_superres[i], ims_gt[i]))
    unet_psnr_list.append(psnr(ims_superres[i], ims_gt[i]))
    unet_ssim_list.append(ssim(ims_superres[i], ims_gt[i], multichannel=True))

print('Mean MSE of interpolation:', np.mean(interp_mse_list))
print('Mean PSNR of interpolation:', np.mean(interp_psnr_list))
print('Mean SSIM of interpolation:', np.mean(interp_ssim_list))
print('Mean MSE of U-net:', np.mean(unet_mse_list))
print('Mean PSNR of U-net:', np.mean(unet_psnr_list))
print('Mean SSIM of U-net:', np.mean(unet_ssim_list))



# Plot reconstruction success.
idx = 14

plt.rcParams.update({'font.size': 22})
fig, axes = plt.subplots(1, 3, figsize=(40, 20))

axes[0].imshow(ims_gt[idx])
axes[0].set_title('Original')
axes[1].imshow(ims_interp[idx])
axes[1].set_title('Interpolated')
axes[2].imshow(ims_superres[idx])
axes[2].set_title('U-Net')
plt.savefig('figures/super-duper{}.png'.format(idx), bbox_inches='tight')
plt.show()