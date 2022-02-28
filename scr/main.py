import torch
import os
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import network_models

print("Cloud GAN Training")
total_epoch = 128
dataset_batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#define transforms
dataset_transformation = transforms.Compose([
    transforms.Resize(64),
    #transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5], 
        [0.5, 0.5, 0.5]
    )#64x3x64x64
])

#load Dataset
image_dataset_folder = torchvision.datasets.ImageFolder(root = "../dataset/", transform = dataset_transformation)
dataset_loader = torch.utils.data.DataLoader(image_dataset_folder, batch_size = dataset_batch_size, shuffle = True)

#Load Model
generator = network_models.Basic_Generator_1().to(device)
generator_optimiser = torch.optim.Adam(generator.parameters(), lr = 0.0002, betas = (0.5, 0.999))

discriminator = network_models.Basic_Discriminator_1().to(device)
discriminator_optimiser = torch.optim.Adam(discriminator.parameters(), lr = 0.0002, betas = (0.5, 0.999))

loss_function = nn.BCEwithLogitsLoss()

#Model Training
for current_epoch in tqdm(range(total_epoch)):
    
    #load dataset
    for i, img in enumerate(dataset_loader):
        
        real_logits = torch.ones(img.shape[0], 2, 1, 1).to(device)
        fake_logits = torch.zeros(img.shape[0], 2, 1, 1).to(device)
        
        #train Generator      
        random_noise = torch.randn(img.shape[0], 5, 1, 1).to(device)
        
        generated_img = generator(random_noise)
        prediction_generated = discriminator(generated_img) 
        gen_loss = loss_function(prediction_generated, real_logits)
        
        gen_loss.backward()
        generator_optimiser.step()
        generator_optimiser.zero_grad()
        
        #train Discriminator
        discriminator_optimiser.zero_grad()
        
        real_img = img.to(device)
        fake_img = generated_img.detach()
        
        predicted_real = discriminator(real_img)
        predicted_fake = discriminator(fake_img)
        
        prediction_real_loss = loss_function(predicted_real, real_logits)
        prediction_fake_loss = loss_function(predicted_fake, fake_logits)
        
        avg_discriminator_loss = (prediction_real_loss + prediction_fake_loss) / 2
        avg_discriminator_loss.backward()
        
        discriminator_optimiser.step()
        
        
        
           


