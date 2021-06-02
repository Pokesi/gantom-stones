import torch
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.utils import make_grid
from PIL import Image
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

def squareifyImage(img, min_size=256, fill_color=(0,0,0,1)):
    x, y = img.size
    size = max(min_size, x, y)
    new_img = Image.new('RGBA', (size, size), fill_color)
    new_img.paste(img, (int((size - x) / 2), int((size - y) / 2)))
    return new_img

for i, filename in enumerate(glob.glob('src/images/**/*.png')):
    print(filename[4:])
    img = squareifyImage(Image.open(filename))
    x = filename.split(sep='/')
    
    os.makedirs('src/newimages/'+x[2], exist_ok=True)
    img.save('src/newimages/{}/{}'.format(x[2],x[3]))

os.getcwd()



