import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

from modelv2 import Discriminator, Generator, weights_init


import time
from tqdm import tqdm

from matplotlib import pyplot as plt




torch.autograd.set_detect_anomaly(True)

BATCH_SIZE = 8
LATENT_DIM = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def return_samples(LABEL, data):
    index = {"F": [51,52,53,65,66,68,118,119,120,121],
             "N": [2,3,4,5,6,7,8,9,10,11],
             "Q": [3,4,5,6,7,8,9,10,12,13],
             "S": [2373,2374,2375,2376,2377,2378,2379,2380,2381,2382],
             "V": [8,9,10,11,12,13,14,15,16,17]}
    
    y = data[index[LABEL]]
    
    return torch.Tensor(y), torch.Tensor(y), torch.Tensor(y)



def batching(a, n):
    n = min(n, len(a))
    k, m = divmod(len(a), n)
    
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def train(LABEL, epochs):
    
    # load the data
    data = pd.read_pickle("data/"+LABEL+".pkl")
    
    data = data.sample(frac=1).reset_index(drop=True)
    
    batches = list(batching(range(len(data)), int(len(data)/BATCH_SIZE)+1))
    X = np.array(data["beat"].to_list())
    X = torch.Tensor(X)
    X.requires_grad = True
    
    netD = Discriminator().to(device)
    netG = Generator(LATENT_DIM).to(device)
    netD.apply(weights_init)
    netG.apply(weights_init)
    
    optimizerD = optim.Adam(netD.parameters(), lr = 0.0001, betas=(0.5,0.999))
    optimizerG = optim.Adam(netG.parameters(), lr = 0.0001, betas=(0.5,0.999))
    # optimizerD = optim.RMSprop(netD.parameters(), lr = 0.0001)
    # optimizerG = optim.RMSprop(netG.parameters(), lr = 0.0001)
    criterion = nn.BCELoss()
    
    running_d_real_loss = []
    running_d_fake_loss = []
    running_g_loss = []
    
    for e in tqdm(range(epochs)):
        
        real_label = 1.
        fake_label = 0.
        

        t1 = np.random.choice(len(batches))
        for b in batches[:50]:
            
            
            netD.zero_grad()
            
            
            real_data = X[b].to(device)
            real_data = real_data.reshape(real_data.size(0), 1, real_data.size(-1))
            
            batch_size = real_data.size(0)
            
            label = torch.full((1,), real_label, device = device)
            output = netD(real_data)
            errD_real = criterion(output, label)

            errD_real.backward()
            
            # D_x = output.mean().item()
            
            
            ## training with fake data
            
            noise = torch.randn((batch_size, LATENT_DIM, 1), device=device)
            fake_data = netG(noise)

            label.fill_(fake_label)
            
            output = netD(fake_data.detach())
            
            
            errD_fake = criterion(output, label)
            errD_fake.backward()
            
            # D_F_z1 = output.mean().item()
            
            errD = errD_real + errD_fake
            optimizerD.step()
            
            
            ## updating generator
            
            netG.zero_grad()
            output = netD(fake_data)
        
            label.fill_(real_label)
            
            
            errG = criterion(output, label)
            errG.backward()
            # D_G_z2 = output.mean().item()
            optimizerG.step()
            
        running_g_loss.append(errG.item())
        running_d_fake_loss.append(errD_fake.item())
        running_d_real_loss.append(errD_real.item())
        
        if (e+1)%200 == 0:
            print(f"Epoch: {e} | Loss_D_real: {errD_real.item()} | Loss_D_fake: {errD_fake.item()}| Loss_G: {errG.item()} | Time: {time.strftime('%H:%M:%S')}")
            static_noise = torch.randn((64, LATENT_DIM, 1)).to(device)
            fake = netG(static_noise.detach())
            plt.plot(fake.detach().cpu().squeeze(1).numpy()[:].transpose())
            plt.savefig("ClassicGAN/plots/"+LABEL+"/"+str(e)+".png")
            plt.close()
            
        if (e+1)%200 == 0:
            torch.save(netG.state_dict(), f"ClassicGAN/saved_models/"+LABEL+"/generator.pth")
            torch.save(netG.state_dict(), f"ClassicGAN/saved_models/"+LABEL+"/discriminator.pth")
            plt.figure(figsize=(10,2))
            plt.plot(running_d_fake_loss, label = "D_fake")
            plt.plot(running_d_real_loss, label = "D_real")
            plt.plot(running_g_loss, label = "G")
            plt.legend()
            plt.savefig("ClassicGAN/plots/"+LABEL+"curves.png")
            plt.close()
            
    
    
    
if __name__ == '__main__':
    
    train("Q", 1000)