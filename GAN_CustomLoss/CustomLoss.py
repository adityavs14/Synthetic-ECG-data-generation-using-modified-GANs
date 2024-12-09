import torch
import torch.nn as nn
import piqa
from tslearn.metrics import SoftDTWLossPyTorch
from utils import utility as utils


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.utils = utils()
        self.ssim = piqa.ssim.SSIM(reduction='none')
        self.psnr = piqa.psnr.PSNR(reduction='none')
        self.haarpsi = piqa.haarpsi.HaarPSI(reduction='none')
        self.dtw = SoftDTWLossPyTorch(gamma=0.01)
        self.bce = nn.BCELoss()

    def forward(self, input, samples, output, labels):
        # Detach inputs and samples for generating images
        input_img = input.detach()
        samples_img = samples.detach()
        
        tmp_img = self.utils.convert(input_img)
        tmp_sample = self.utils.convert(samples_img)

        # Convert and permute images in a batch-wise manner
        img = torch.tensor(tmp_img, dtype=torch.float32).permute(0, 3, 1, 2)
        sample_img = torch.tensor(tmp_sample, dtype=torch.float32).permute(0, 3, 1, 2)
        
        

        batch_size, num_samples = input.shape[0], samples.shape[0]

        # Prepare metrics tensor
        metrics = torch.zeros((batch_size, 2, num_samples), device=input.device)

        # SSIM, PSNR, and HaarPSI are computed for all samples in a batch-wise manner
        for i in range(num_samples):
            metrics[:, 0, i] = (1 - self.ssim(sample_img[i:i + 1], img)) * 10
        #     metrics[:, 1, i] = -self.psnr(sample_img[i:i + 1], img) / 100
        #     metrics[:, 2, i] = (1 - self.haarpsi(sample_img[i:i + 1], img)) * 10

        # DTW is computed separately due to its structure
        for i in range(num_samples):
            metrics[:, 1, i] = self.dtw(
                input.reshape(batch_size, -1, 1),
                samples[i].reshape(1, -1, 1).expand(batch_size, -1, -1)
            ) / 500

        # Average metrics over samples and reduce further over dimensions
        metrics = metrics.mean(dim=2).mean(dim=1)

        # Compute BCE loss
        bce_loss = self.bce(output, labels)

        # Combine losses (normalization can be added if necessary)
        loss = bce_loss + metrics.sum()/10

        return loss, (bce_loss.item(), metrics.sum().item())