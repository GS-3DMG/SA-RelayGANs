import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from spectral import SpectralNorm
from einops import rearrange
import numpy as np

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention


class Generator(nn.Module):
    """Generator."""

    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64):
        super(Generator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        layer5 = []
        last = []

        con1 = []
        con2 = []
        con3 = []
        con4 = []
        con5 = []
        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num # 8
        base_dim = conv_dim * mult
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, base_dim, 4)))
        layer1.append(nn.BatchNorm2d(base_dim))
        layer1.append(nn.ReLU())
        con_dim = base_dim // 8
        con1.append(self.cond_block(2 * 2**1, cond_dim=con_dim))
        curr_dim = base_dim + con_dim
        setattr(self, f'layer1', layer1)
        setattr(self, f'con1', con1)

        base_dim = int(base_dim / 2)
        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, base_dim, 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(base_dim))
        layer2.append(nn.ReLU())
        con_dim = base_dim // 8
        con2.append(self.cond_block(2 * 2**2, cond_dim=con_dim))
        curr_dim = base_dim + con_dim
        setattr(self, f'layer2', layer2)
        setattr(self, f'con2', con2)

        base_dim = int(base_dim / 2)
        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, base_dim, 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(base_dim))
        layer3.append(nn.ReLU())
        con_dim = base_dim // 8
        con3.append(self.cond_block(2 * 2 ** 3, cond_dim=con_dim))
        curr_dim = base_dim + con_dim
        setattr(self, f'layer3', layer3)
        setattr(self, f'con3', con3)

        base_dim = int(base_dim / 2)
        layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, base_dim, 4, 2, 1)))
        layer4.append(nn.BatchNorm2d(base_dim))
        layer4.append(nn.ReLU())
        con_dim = base_dim // 8
        con4.append(self.cond_block(2 * 2 ** 4, cond_dim=con_dim))
        curr_dim = base_dim + con_dim
        setattr(self, f'layer4', layer4)
        setattr(self, f'con4', con4)

        base_dim = int(base_dim / 2)
        layer5.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, base_dim, 4, 2, 1)))
        layer5.append(nn.BatchNorm2d(base_dim))
        layer5.append(nn.ReLU())
        con_dim = base_dim // 8 
        con5.append(self.cond_block(2 * 2 ** 5, cond_dim=con_dim))
        curr_dim = base_dim + con_dim
        setattr(self, f'layer5', layer5)
        setattr(self, f'con5', con5)


        #
        # if self.imsize == 64:
        #     layer4 = []
        #     curr_dim = int(curr_dim / 2)
        #     layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        #     layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
        #     layer4.append(nn.ReLU())
        #     self.l4 = nn.Sequential(*layer4)
        #     curr_dim = int(curr_dim / 2)
        # elif self.imsize == 128:
        #     layer5 = []
        #     curr_dim = int(curr_dim / 2)
        #     # layer5.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        #     layer5.append(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1))
        #     layer5.append(nn.BatchNorm2d(int(curr_dim / 2)))
        #     layer5.append(nn.ReLU())
        #     self.l5 = nn.Sequential(*layer5)
        #     curr_dim = int(curr_dim / 2)
        #
        self.l1 = nn.Sequential(*layer1)
        self.c1 = nn.Sequential(*con1)
        self.l2 = nn.Sequential(*layer2)
        self.c2 = nn.Sequential(*con2)
        self.l3 = nn.Sequential(*layer3)
        self.c3 = nn.Sequential(*con3)
        self.l4 = nn.Sequential(*layer4)
        self.c4 = nn.Sequential(*con4)
        self.l5 = nn.Sequential(*layer5)
        self.c5 = nn.Sequential(*con5)

        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(288, 'relu')
        self.attn2 = Self_Attn(144,  'relu')

    def cond_block(self, out, cond_dim) -> nn.Module:
        class CondModule(nn.Module):
            def __init__(self, in_size: int, in_dim: int, out_size: int, out_dim: int) -> None:
                super().__init__()
                self.in_size = in_size
                self.out_size = out_size
                self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=1)

            def forward(self, pixel_maps: torch.Tensor) -> torch.Tensor:
                pixel_indic = pixel_maps[:, 0:1]
                pixel_cls = pixel_maps[:, 1:]
                kernel_size = self.in_size // self.out_size
                resized_indic = nn.MaxPool2d(kernel_size=kernel_size,stride=kernel_size)(pixel_indic)
                resized_cls = nn.AvgPool2d(kernel_size=kernel_size,
                                           stride=kernel_size)(pixel_cls)
                resized_maps = torch.cat([resized_indic, resized_cls], dim=1)
                out = self.conv(resized_maps)
                return out

        return CondModule(self.imsize, 3, out, cond_dim)

    def forward(self, z, cd):
        z = z.view(z.size(0), z.size(1), 1, 1)
        # x = rearrange(z, '(B h w) z_dim -> B z_dim h w', h=1, w=1)

        # out = getattr(self, f'layer1')(z)
        # out_c = getattr(self, f'con1')(cd)
        # out = torch.cat([out, out_c], dim=1)
        # out = getattr(self, f'layer2')(out)
        # out_c = getattr(self, f'con2')(cd)
        # out = torch.cat([out, out_c], dim=1)
        # out = getattr(self, f'layer3')(out)
        # out_c = getattr(self, f'con3')(cd)
        # out = torch.cat([out, out_c], dim=1)
        # out = getattr(self, f'layer4')(out)
        # out_c = getattr(self, f'con4')(cd)
        # out = torch.cat([out, out_c], dim=1)
        # out, p1 = self.attn1(out)
        # out = getattr(self, f'layer5')(out)
        # out_c = getattr(self, f'con5')(cd)
        # out = torch.cat([out, out_c], dim=1)
        # out, p2 = self.attn2(out)
        out=self.l1(z)
        out_c = self.c1(cd)
        out = torch.cat([out, out_c], dim=1)
        out=self.l2(out)
        out_c = self.c2(cd)
        out = torch.cat([out, out_c], dim=1)
        out=self.l3(out)
        out_c = self.c3(cd)
        out = torch.cat([out, out_c], dim=1)
        out=self.l4(out)
        out_c = self.c4(cd)
        out = torch.cat([out, out_c], dim=1)
        out, p1 = self.attn1(out)
        out=self.l5(out)
        out_c = self.c5(cd)
        out = torch.cat([out, out_c], dim=1)
        out,p2 = self.attn2(out)
        out=self.last(out)
        return out, p1, p2


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        # layer1.append(nn.Conv2d(1, conv_dim, 4, 2, 1))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        # layer2.append(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        # layer3.append(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        # layer4.append(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1))
        layer4.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        if self.imsize == 128:
            layer5 = []
            layer5.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            # layer5.append(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1))
            layer5.append(nn.LeakyReLU(0.1))
            self.l5 = nn.Sequential(*layer5)
            curr_dim = curr_dim*2

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(1024, 'relu')
        self.attn2 = Self_Attn(2048, 'relu')

    def forward(self, x):
        # print(x.shape)
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out,p1 = self.attn1(out)
        out=self.l5(out)
        out,p2 = self.attn2(out)
        out=self.last(out)

        return out.squeeze(), p1, p2
