import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

from modelv2 import Discriminator, Generator, weights_init
from CustomLoss import CustomLoss


import time
from tqdm import tqdm

from matplotlib import pyplot as plt




torch.autograd.set_detect_anomaly(True)

BATCH_SIZE = 8
LATENT_DIM = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

def return_samples(LABEL, data):
    index = {"F": [a for a in range(417,712)],
             "N": [a for a in range(2,1002)],
             "Q": [a for a in range(46,1133)],
             "S": [a for a in range(1953, 2721)],
             "V": [a for a in range(854, 1295)]}
    
    y = data[index[LABEL]]
    y = y.reshape(y.shape[0], 1, y.shape[-1])
    
    return torch.tensor(y), torch.tensor(y)



def batching(a, n):
    n = min(n, len(a))
    k, m = divmod(len(a), n)
    
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def train(LABEL, epochs):
    

    
    # load the data
    data = pd.read_pickle("data/"+LABEL+".pkl")
    
    tmp = np.array(data["beat"].to_list())
    # tmp = sc.fit_transform(tmp.T).T

    sample_real_d, sample_real_g = return_samples(LABEL,tmp)
    
    del tmp
    
    data = data.sample(frac=1).reset_index(drop=True)
    
    batches = list(batching(range(len(data)), int(len(data)/BATCH_SIZE)+1))
    X = np.array(data["beat"].to_list())
    X = torch.Tensor(X)
    X.requires_grad = True

    
    netD = Discriminator().to(device)
    netG = Generator(LATENT_DIM).to(device)
    netD.apply(weights_init)
    netG.apply(weights_init)
    
    optimizerD = optim.Adam(netD.parameters(), lr = 0.0001, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr = 0.0001, betas=(0.5, 0.999))
    criterion = CustomLoss()
    BCEcriterion = nn.BCELoss()
    
    static_noise = torch.randn((BATCH_SIZE, LATENT_DIM, 1)).to(device)
    
    
    running_d_fake_loss = []
    running_d_real_loss = []
    running_g_loss = []
    
    for e in tqdm(range(epochs)):
        
        real_label = 1.
        fake_label = 0.
        
        
    
        for b in batches[:50]:
            
            netD.zero_grad()
            
            real_data_d = X[b].to(device)
            real_data_d = real_data_d.reshape(real_data_d.size(0), 1, real_data_d.size(-1))
            batch_size = real_data_d.size(0)
            
            
            label_real_d  = torch.full((1,), real_label, device = device)
            output_real_d = netD(real_data_d)
            errD_real = BCEcriterion(output_real_d.cpu(), label_real_d.cpu())
            
            errD_real.backward()

            
            ## training with fake data
            
            noise = torch.randn((batch_size, LATENT_DIM, 1), device=device)
            fake_data_d = netG(noise)
            label_fake_d = torch.full((1,), fake_label, device = device)

            
            output_fake_d = netD(fake_data_d.detach())
            errD_fake = BCEcriterion(output_fake_d.cpu(), label_fake_d.cpu())
            errD_fake.backward()
            
            
            errD = errD_real + errD_fake
            optimizerD.step()
            
            
            ## updating generator
            netG.zero_grad()
            output_fake_g = netD(fake_data_d)
            label_real_g = torch.full((1,), real_label, device = device)
    
            
            errG, loss3 = criterion(fake_data_d.cpu(), sample_real_g[np.random.choice(sample_real_g.shape[0])].reshape(1, 1, 280).detach().cpu(),
                            output_fake_g.cpu(), label_real_g.cpu())
            errG.backward()
            optimizerG.step()
            
        running_g_loss.append(errG.item())
        running_d_fake_loss.append(errD_fake.item())
        running_d_real_loss.append(errD_real.item())
        
        if (e+1)%100 == 0:
            print(f"Epoch: {e} | Loss_D_fake: {errD_fake.item()} | Loss_D_real: {errD_real.item()} | Loss_G: {errG.item()} | Loss_metric: {loss3[1]} ")
            fake = netG(static_noise)
            plt.plot(fake.detach().cpu().squeeze(1).numpy()[:].transpose())
            plt.savefig("GAN_CustomLoss/plots2d/"+LABEL+"/"+str(e)+".png")
            plt.close()
            
        if (e+1)%100 == 0:
            torch.save(netG.state_dict(), f"GAN_CustomLoss/saved_models_2d/"+LABEL+"/generator.pth")
            torch.save(netG.state_dict(), f"GAN_CustomLoss/saved_models_2d/"+LABEL+"/discriminator.pth")
            plt.figure(figsize = (10,4))
            plt.plot(running_d_fake_loss, label = "D_fake")
            plt.plot(running_d_real_loss, label = "D_real")
            plt.plot(running_g_loss, label = "G")
            plt.legend()
            plt.savefig("GAN_CustomLoss/plots2d/"+LABEL+"curves.png")
            plt.close()
    
    
if __name__ == '__main__':
    
    train("N", 300)