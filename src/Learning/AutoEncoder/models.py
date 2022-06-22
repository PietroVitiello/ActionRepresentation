import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class SpatialSoftArgmax(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x):
        # print(f"x inside activation: {x.shape}")
        x = torch.div(x, self.alpha)
        # print(f"x after division: {x.shape}")
        x = nn.Softmax2d()(x)
        # print(f"x after softmax: {x.shape}")
        x_x = torch.mean(torch.sum(x, dim=3), dim=2)
        x_y = torch.mean(torch.sum(x, dim=2), dim=2)
        # print(f"x_x: {x_x.shape}")
        # print(f"x_y: {x_y.shape}")
        return torch.cat((x_x, x_y), dim=1)



class SpatialAE_fc(nn.Module):
    def __init__(
        self,
        num_outputs=6,
        num_aux_outputs=9,
        reconstruction_size=16
    ):
        super(SpatialAE_fc,self).__init__()
        self.recon_size = reconstruction_size
        self.spatialArgmax = SpatialSoftArgmax()

        self.encoder = self.Encoder()
        self.decoder = self.Decoder(reconstruction_size)

        self.fc1 = nn.Sequential(nn.Linear(256, 128),
                                 nn.ReLU())

        self.aux = nn.Linear(128, num_aux_outputs)
        self.out = nn.Linear(128, num_outputs)

    def Encoder(self):
        encoder = nn.Sequential(
            self.conv_layer(3, 32),
            self.conv_layer(32, 64),
            self.conv_layer(64, 128),
            self.spatialArgmax
        )
        return encoder

    def Decoder(self, img_size: int):
        decoder = nn.Linear(256, img_size**2)
        return decoder

    def decoder_forward(self, x: torch.Tensor):
        self.decoder(x)
        return x.view(x.size(0), self.recon_size, -1)

    def conv_layer(
        self,
        chIN,
        chOUT,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        pool_kernel=3,
        pool_stride=2,
        pool_padding=1
    ):
        conv = nn.Sequential(nn.Conv2d(chIN, chOUT, kernel_size, stride, padding, bias=bias),
                             nn.BatchNorm2d(chOUT),
                             nn.MaxPool2d(pool_kernel, pool_stride, pool_padding),
                             nn.ReLU(inplace=True))
        return conv

    def forward(self, x):
        # print(x.shape)
        x: torch.Tensor = self.encoder(x) 
        # print(f"x after activation: {x.shape}")       
        fc1 = self.fc1(x)

        if self.training:
            recon_img = self.decoder_forward(x)
            x_aux = self.aux(fc1)
            x_out = self.out(fc1)
            out = torch.cat((x_out, x_aux), dim=1)
            return (recon_img, out)
        else:
            return self.out(fc1)