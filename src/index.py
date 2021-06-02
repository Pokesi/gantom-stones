import torch.nn as nn
import torch
from torchvision.utils import save_image
import torchvision
import os
from flask import Flask, send_file
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import io
import random

def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]

stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
device = torch.device('cpu')
latent_size = 256

app = Flask(__name__)


@app.route('/')
def index():
    x = random.randrange(0,4)
    
    fixed_latent = torch.randn(64, latent_size, 1, 1, device=torch.device('cpu')) 
    
    if x==0:
        generator = torch.load('saved_models/generator_model-0.pth', map_location=device)
    elif x==1:
        generator = torch.load('saved_models/generator_model-1.pth', map_location=device)
    elif x==2:
        generator = torch.load('saved_models/generator_model-4.pth', map_location=device)
    else:
        generator = torch.load('saved_models/generator_model-3.pth', map_location=device)


    fake_images = generator(fixed_latent)
        
    img_pil = torchvision.transforms.functional.to_pil_image(denorm(fake_images[0]))

    im = img_pil
    im_resize = im.resize((256,256))
    buf = io.BytesIO()
    im_resize.save(buf, format='PNG')
    byte_im = buf.getvalue()

    return send_file(
        io.BytesIO(byte_im),
        mimetype='image/png',
        as_attachment=False,
        attachment_filename='tst.png'), str(x)

