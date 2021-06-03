import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.utils import make_grid
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm
import torch.nn.functional as F
import spell.metrics as metrics
import discord
from discord import Webhook, RequestsWebhookAdapter
from random import randrange


def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]


batch_size = 4
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

os.getcwd()

img_dataset = ImageFolder('src/images/', transform=T.Compose([
    T.Pad(70),
    T.Resize((256,256)),
    T.ToTensor(),
    T.Normalize(*stats)
]
))


train_dl = DataLoader(img_dataset, batch_size, shuffle=True,
                      num_workers=2, pin_memory=True)
#train_dl = DataLoader(img_dataset, batch_size, shuffle=True)


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(16,16))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(
        make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))


def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break


# +
#show_batch(train_dl)
# -


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)

discriminator = nn.Sequential(
    # in: 3 x 256 x 256

    nn.Conv2d(3, 64, kernel_size=16, stride=4, padding=4, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 64 x 32 x 32

    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 128 x 16 x 16

    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 256 x 8 x 8

    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 512 x 4 x 4

    nn.Conv2d(512, 1, kernel_size=7, stride=1, padding=0, bias=False),
    # out: 1 x 1 x 1

    nn.Flatten(),
    nn.Sigmoid())


# +
discriminator = to_device(discriminator, device)

latent_size = 256
# -

generator = nn.Sequential(
    # in: latent_size x 1 x 1

    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    # out: 512 x 4 x 4

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    # out: 256 x 8 x 8

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # out: 128 x 16 x 16

    nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # out: 64 x 32 x 32

    nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # out: 32 x 64 x 64
    
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    # out: 16 x 128 x 128
    
    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
    # out: 3 x 256 x 256
)

xb = torch.randn(batch_size, latent_size, 1, 1) # random latent tensors
fake_images = generator(xb)
generator = to_device(generator, device)
#print(fake_images.size())


def train_discriminator(real_images, opt_d):
    # Clear discriminator gradients
    opt_d.zero_grad()

    # Pass real images through discriminator
    real_preds = discriminator(real_images)
    min_real, max_real = 0, 0.1
    real_targets = (max_real-min_real)*torch.rand(real_images.size(0), 1, device=device) + min_real
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()

    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Pass fake images through discriminator
    min_fake, max_fake = 0.9, 1
    fake_targets = (max_fake-min_fake)*torch.rand(fake_images.size(0), 1, device=device) + min_fake
    fake_preds = discriminator(fake_images)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score


def train_generator(opt_g):
    # Clear generator gradients
    opt_g.zero_grad()

    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Try to fool the discriminator
    preds = discriminator(fake_images)
    min, max = 0, 0.1
    #targets = torch.zeros(batch_size, 1, device=device)
    targets = (max-min)*torch.rand(batch_size, 1, device=device) + min
    loss = F.binary_cross_entropy(preds, targets)

    # Update generator weights
    loss.backward()
    opt_g.step()
    return loss.item()


sample_dir = 'generated'
os.makedirs(sample_dir, exist_ok=True)

webhook = Webhook.partial('846994789713838091', 'A_PyxEqc5vxd_eJm6p99Vu6awYo2n-VXn6gld_65n41iWfmFWdIrkD6w4l0_9Ep5MvVm',\
adapter=RequestsWebhookAdapter())


# +
def save_samples(index, latent_tensors, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    image = os.path.join(sample_dir, fake_fname)
    x= randrange(0,63)
    save_image(denorm(fake_images[x]), image)

    try:
        webhook.send(file=discord.File(image))
    except:
        print('skipping webhook')
    
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))

fixed_latent = torch.randn(8, latent_size, 1, 1, device=device)
i = 0

# -

def fit(epochs, lr, start_idx=1):
    torch.cuda.empty_cache()

    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=.0002, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for n, (real_images, _) in enumerate(tqdm(train_dl)):

            if n%10==0:
                metrics.send_metric('Batch',n)

            # Train discriminator
            loss_d, real_score, fake_score = train_discriminator(real_images, opt_d)
            # Train generator
            loss_g = train_generator(opt_g)

        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        # Log losses & scores (last batch)
        metrics.send_metric('Epoch', epoch+1 )

        output = "Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            i, epochs, loss_g, loss_d, real_score, fake_score)
        # print(output)

        # Save generated images
        if epoch%10==0: 
            
            webhook.send(output)
            save_samples(epoch+start_idx, fixed_latent, show=False)

            os.makedirs('model', exist_ok=True)
            torch.save(generator.state_dict(), 'model/model-weights-{}.pt'.format(i))
            torch.save(generator, 'saved_models/generator_model-{}.pt'.format(i))
            i+=1
            

    return losses_g, losses_d, real_scores, fake_scores


lr = 0.0001
epochs = 1000

history = fit(epochs, lr)




