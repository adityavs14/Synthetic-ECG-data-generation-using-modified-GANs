import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(

            nn.Conv1d(1, 32, 4, 4, 1, bias=False),
            nn.Dropout(0.2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(32, 64, 4, 4, 1, bias=False),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 1, 16, 4, 0, bias=False),
            nn.Dropout(0.2),
            nn.Sigmoid()
        )
        # self.model = nn.Sequential(
        #     nn.Conv1d(1, 32, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),

        #     nn.Conv1d(32, 64, 4, 2, 1, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.LeakyReLU(0.2, inplace=True),
            
        #     nn.Conv1d(64, 128, 4, 2, 1, bias=False),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
            
            
        #     nn.Conv1d(128, 256, 4, 2, 1, bias=False),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(0.2, inplace=True),
            
            
        #     nn.Conv1d(256, 512, 4, 2, 1, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(0.2, inplace=True),
            
            
        #     nn.Conv1d(512, 1024, 4, 2, 1, bias=False),
        #     nn.BatchNorm1d(1024),
        #     nn.LeakyReLU(0.2, inplace=True),
            
            
        #     nn.Conv1d(1024, 2048, 4, 2, 1, bias=False),
        #     nn.BatchNorm1d(2048),
        #     nn.LeakyReLU(0.2, inplace=True),

        #     nn.Conv1d(2048, 1, 4, 2, 1, bias=False),
        #     nn.Sigmoid()
        # )
        # self.lstm = nn.LSTM(
        #         input_size=280,
        #         hidden_size=128,
        #         num_layers=1,
        #         bidirectional=True,
        #         batch_first=True,
        #     )

    def forward(self, x):
        # x = self.lstm(x)[0]
        x = self.model(x).view(-1)
        return x



class MetricDiscriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.LeakyReLU(0.2,True),
            
            nn.Linear(16,8),
            nn.LeakyReLU(0.2,True),
            
            nn.Linear(8,4),
            nn.LeakyReLU(0.2, True),
            
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x).view(-1)
        


class Generator(nn.Module):
    def __init__(self,nz):
        super().__init__()
        # self.model = nn.Sequential(
        #     nn.ConvTranspose1d(512, 2048, 20, 2, 1, bias=False),
        #     nn.BatchNorm1d(2048),
        #     nn.LeakyReLU(0.2, True),

        #     nn.ConvTranspose1d(2048, 1024, 20, 2, 1, bias=False),
        #     nn.BatchNorm1d(1024),
        #     nn.LeakyReLU(0.2,True),

        #     nn.ConvTranspose1d(1024, 512, 20, 2, 1, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(0.2,True),

        #     nn.ConvTranspose1d(512, 256, 20, 2, 1, bias=False),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(0.2,True),
            
        #     nn.ConvTranspose1d(256, 128, 20, 2, 1, bias=False),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(0.2,True),
            
        #     nn.ConvTranspose1d(128, 64, 20, 2, 1, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.LeakyReLU(0.2,True),

        #     nn.ConvTranspose1d(64, 1, 20, 5, 0, bias=False),
        #     # nn.Sigmoid() # input data is not normalized
        # )
        self.model = nn.Sequential(
            nn.ConvTranspose1d(nz, 512, 16, 2, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose1d(512, 256, 8, 2, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2,True),

            nn.ConvTranspose1d(256, 128, 8, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2,True),

            nn.ConvTranspose1d(128, 64, 8, 2, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2,True),
            
            nn.ConvTranspose1d(64, 1, 8, 2, 1, bias=False),
        )
        self.lstm = nn.LSTM(
                input_size=nz,
                hidden_size=256,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )

    def forward(self, x):
        # x = x.reshape(x.shape[0], x.shape[2], x.shape[1])
        # x = self.lstm(x)[0]
        # x = x.reshape(x.shape[0], x.shape[2], x.shape[1])
        x = self.model(x)
        return x[: ,:, :280]