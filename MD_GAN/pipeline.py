import cv2 as cv
from metrics import *
import matplotlib.pyplot as plt
import torch
from utils import utility
from tslearn.metrics import SoftDTWLossPyTorch


class pipeline:
    def __init__(self, metric_size):
        self.utils = utility()
        self.dtw = SoftDTWLossPyTorch(gamma=0.01)
        self.metric_size = metric_size
        
    def process(self, input, samples):
        # Convert images once, not repeatedly
        input_imgs = self.utils.convert(input.squeeze(1))
        sample_imgs = self.utils.convert(samples.squeeze(1))

        batch_size, sample_size = input_imgs.shape[0], sample_imgs.shape[0]
        
        # Pre-allocate metrics tensor
        metrics = torch.zeros((batch_size, self.metric_size, sample_size), device=input.device)
        
        # Vectorize outer loop
        for s in range(sample_size):
            grayB = sample_imgs[s]  # Shared across batch
            sigB = samples[s].reshape(1, -1, 1)  # Shared across batch
            
            # Parallelize batch computation
            grayA = input_imgs  # All input images
            sigA = input.reshape(batch_size, -1, 1)  # All input signals
            
            # Precompute DTW for the entire batch
            dtw_values = torch.stack([self.dtw(sigA[b].reshape(1,-1,1), sigB) for b in range(batch_size)])
            
            # Compute all metrics for the batch
            metric_results = torch.stack([torch.tensor([UQI.process(grayA[b], grayB),
                                          VIFP.process(grayA[b], grayB),
                                          SCC.process(grayA[b], grayB),
                                          SAM.process(grayA[b], grayB),
                                          ERGAS.process(grayA[b], grayB),
                                          RASE.process(grayA[b], grayB),
                                          SIFT.process(grayA[b], grayB),
                                          SSIM.process(grayA[b], grayB),
                                          dtw_values[b]]) for b in range(batch_size)], dim=0)
            
            metrics[:, :, s] = metric_results

        # Take the mean across the sample dimension
        metrics = metrics.mean(dim=2)
        return metrics
        
        
    # def process(self, input, samples):
        
    #     input_imgs = self.utils.convert(input.squeeze(1))
    #     sample_imgs = self.utils.convert(samples.squeeze(1))
        
    #     batch_size, sample_size = input_imgs.shape[0], sample_imgs.shape[0]
        
    #     metrics = torch.zeros((batch_size, self.metric_size, sample_size), device=input.device)
        
    #     for s in range(sample_size):
    #         for b in range(batch_size):
    #             metrics[b,:,s] = self.metric_calc(input_imgs[b], sample_imgs[s], input[b], samples[b])
                
    #     metrics = metrics.mean(dim=2)
    #     return metrics
    
    # def metric_calc(self, grayA, grayB, sigA, sigB):
    #     sigA = sigA.reshape(1, -1,1)
    #     sigB = sigB.reshape(1, -1,1)
    #     dtw = self.dtw(sigA,sigB)
    #     return torch.tensor([UQI.process(grayA,grayB),
    #             VIFP.process(grayA,grayB),
    #             SCC.process(grayA,grayB),
    #             SAM.process(grayA,grayB),
    #             ERGAS.process(grayA,grayB),
    #             RASE.process(grayA,grayB),
    #             SIFT.process(grayA,grayB),
    #             SSIM.process(grayA,grayB),
    #             dtw])
    

        