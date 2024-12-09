import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

from modelv2 import Discriminator, MetricDiscriminator, Generator, weights_init
from pipeline import pipeline


import time
from tqdm import tqdm

from matplotlib import pyplot as plt



torch.autograd.set_detect_anomaly(True)

BATCH_SIZE = 8
LATENT_DIM = 200
METRICS = 9


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def return_samples(LABEL, data):
    index = {"F": [a for a in range(417,712)],
             "N": [a for a in range(2,1002)],
             "Q": [a for a in range(46,1133)],
             "S": [a for a in range(1953, 2721)],
             "V": [a for a in range(854, 1295)]}
    
    y = data[index[LABEL]]
    y = y.reshape(y.shape[0], 1, y.shape[-1])
    
    return torch.tensor(y), torch.tensor(y)


# def return_samples(LABEL, data):
#     # index = {"F": [51],
#     #          "N": [2],
#     #          "Q": [3],
#     #          "S": [2373],
#     #          "V": [8]}
    
#     # y = data[index[LABEL]]
#     y = data.reshape(data.shape[0], 1, data.shape[-1]) #sending back the entire real_data
    
#     return torch.tensor(y), torch.tensor(y)

def batching(a, n):
    n = min(n, len(a))
    k, m = divmod(len(a), n)
    
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False
        
def unfreeze_params(model):
    for param in model.parameters():
        param.requires_grad = True


def train(LABEL, epochs):
    
    
    # loading the data
    data = pd.read_pickle("data/"+LABEL+".pkl")
    
    tmp = np.array(data["beat"].to_list())
    # tmp = sc.fit_transform(tmp.T).T

    samples_real_d, samples_fake_d = return_samples(LABEL,tmp)
    
    del tmp
    
    data = data.sample(frac=1).reset_index(drop=True)
    
    batches = list(batching(range(len(data)), int(len(data)/BATCH_SIZE)+1))
    X = np.array(data["beat"].to_list())
    X = torch.Tensor(X)
    X.requires_grad = True
    
    
    netD = Discriminator().to(device)
    netMD = MetricDiscriminator(METRICS).to(device)
    netG = Generator(LATENT_DIM).to(device)
    
    
    # if loading weights dont apply initialization
    
    netD.apply(weights_init)
    netMD.apply(weights_init)
    netG.apply(weights_init)
    
    # netD.load_state_dict(torch.load("DCGAN/saved_models/"+LABEL+"/discriminator.pth" ,weights_only=True))
    # netMD.load_state_dict(torch.load("DCGAN/saved_models/"+LABEL+"/metric.pth" ,weights_only=True))
    # netG.load_state_dict(torch.load("DCGAN/saved_models/"+LABEL+"/generator.pth" ,weights_only=True))
    
    optimizerD = optim.Adam(netD.parameters(), lr = 0.0001, betas = (0.5,0.999))
    optimizerG = optim.Adam(netG.parameters(), lr = 0.0001, betas = (0.5,0.999))
    optimizerMD = optim.Adam(netMD.parameters(), lr = 0.00001, betas = (0.5,0.999))
    criterion = nn.BCELoss()
    pip = pipeline(METRICS)
    
    # static_noise = torch.randn((128, LATENT_DIM, 1)).to(device)
    
    
    running_d_real_loss = []
    running_d_fake_loss = []
    running_g_fake_loss = []
    running_g_metric_loss = []
    running_m_real_loss = []
    running_m_fake_loss = []
    
    
    for e in tqdm(range(epochs)):
        
        real_label = 1.
        fake_label = 0.
        
        
        t1 = np.random.choice(len(batches))
        for b in batches[t1:t1+5]:
            
            netD.zero_grad()
            netMD.zero_grad()
            
            # for p in netD.parameters():
            #     p.data.clamp_(-0.01, 0.01)
            
            
            real_data_d = X[b].to(device)
            real_data_d = real_data_d.reshape(real_data_d.size(0), 1, real_data_d.size(-1))
            batch_size = real_data_d.size(0)
            
            real_metric = pip.process(real_data_d.detach().cpu(),
                                      samples_real_d[np.random.choice(samples_real_d.shape[0])].reshape(1, 1, 280).detach().cpu()).to(device) # instead of just one base signal we sample from the real dataset
            
            output = netD(real_data_d)
            label = torch.full((batch_size,), real_label, device = device)
            errD_real = criterion(output.cpu(), label.cpu())
            errD_real.backward()
            
            output_metric = netMD(real_metric)
            label = torch.full((batch_size,), real_label, device = device)
            errMD_real = criterion(output_metric.cpu(), label.cpu())
            errMD_real.backward()
            
            
            noise = torch.randn((batch_size, LATENT_DIM, 1), device=device)
            fake_data = netG(noise)
            fake_metric = pip.process(fake_data.detach().cpu(), 
                                      samples_fake_d[np.random.choice(samples_fake_d.shape[0])].reshape(1, 1, 280).detach().cpu()).to(device) # instead of just one base signal we sample from the real dataset
            label = torch.full((batch_size,), fake_label, device = device)
            output = netD(fake_data.detach())
            errD_fake = criterion(output.cpu(), label.cpu())
            errD_fake.backward()
            
            
            label = torch.full((batch_size,), fake_label, device = device)
            output_metric = netMD(fake_metric.detach())
            errMD_fake = criterion(output_metric.cpu(), label.cpu())
            errMD_fake.backward()
            
            # errD = errD_real + errD_fake
            # errM = errMD_real + errMD_fake
            
            optimizerD.step()
            optimizerMD.step()
            
            
            # generator
            
            
            netG.zero_grad()

            label = torch.full((batch_size,), real_label, device = device)
            output = netD(fake_data)
            errG_fake = criterion(output.cpu(), label.cpu())
            
            output_metric = netMD(fake_metric)
            label_metric = torch.full((batch_size,), real_label, device = device)
            errG_metric = criterion(output_metric.cpu(), label_metric.cpu())
            
            errG = errG_fake + errG_metric
            
            
            errG.backward()
            optimizerG.step()
                    
            
                
            
            
            
        running_g_fake_loss.append(errG_fake.item())
        running_g_metric_loss.append(errG_metric.item())
        running_m_real_loss.append(errMD_real.item())
        running_m_fake_loss.append(errMD_fake.item())
        running_d_real_loss.append(errD_real.item())
        running_d_fake_loss.append(errD_fake.item())
        
        if (e+1)%1 == 0:
            # print("BCE (d,g): ", np.mean(BCE_d),np.mean(BCE_g))
            # print("DTW (d,g): ", np.mean(DTW_d),np.mean(DTW_g))
            # print("\n\n\n")
            print(f"Epoch: {e} | Loss_D_real: {errD_real.item()} | Loss_D_fake: {errD_fake.item()} | Loss_M_real: {errMD_real.item()} | Loss_M_fake: {errMD_fake.item()} | Loss_G_fake: {errG_fake.item()} | Loss_G_metric: {errG_metric.item()}| ")
            static_noise = torch.randn((128, LATENT_DIM, 1)).to(device)
            fake = netG(static_noise.detach())
            plt.plot(fake.detach().cpu().squeeze(1).numpy()[:].transpose())
            plt.savefig("MD_GAN/plots/"+LABEL+"/"+str(e)+".png")
            plt.close()
            
        if (e+1)%1 == 0:
            torch.save(netG.state_dict(), f"MD_GAN/saved_models/"+LABEL+"/generator"+str(e+1)+".pth")
            torch.save(netMD.state_dict(), f"MD_GAN/saved_models/"+LABEL+"/metric"+str(e+1)+".pth")
            torch.save(netD.state_dict(), f"MD_GAN/saved_models/"+LABEL+"/discriminator"+str(e+1)+".pth")
            plt.figure(figsize = (10,4))
            plt.plot(running_d_fake_loss, label = "D_fake")
            plt.plot(running_d_real_loss, label = "D_real")
            plt.plot(running_m_fake_loss, label = "M_fake")
            plt.plot(running_m_real_loss, label = "M_real")
            plt.plot(running_g_fake_loss, label = "G_fake")
            plt.plot(running_g_metric_loss, label = "G_metric")
            plt.ylim(-1,2)
            plt.legend()
            plt.savefig("MD_GAN/plots/"+LABEL+"curves.png")
            plt.close()
            
            


if __name__ == '__main__':
    
    train("F", 1)