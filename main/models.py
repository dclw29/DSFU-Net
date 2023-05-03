"""
Adapted from: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/models.py
Include some self-attn into discriminator and generator, and spectral normalisation elsewhere for more stable training
https://towardsdatascience.com/building-your-own-self-attention-gans-e8c9b9fe8e51
"""

import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter
import numpy as np

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

##############################
#           U-NET
##############################

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, spectral_norm=False, dropout=0.0):
        """
        Include spectral norm from below (as bool parameter)
        """
        super(UNetDown, self).__init__()
        if spectral_norm:
            layers = [SpectralNorm(nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False))]
        else:
            layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, spectral_norm=False, dropout=0.0):
        super(UNetUp, self).__init__()
        if spectral_norm:
            layers = [SpectralNorm(nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False))]
        else:
            layers = [nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False)]
        layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.ReLU(inplace=True))

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5, normalize=False)
        #self.down7 = UNetDown(512, 512, dropout=0.5)
        #self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # latent space is 4x4

        #self.up1 = UNetUp(512, 512, dropout=0.5)
        #self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(512, 512, dropout=0.5) # 1024
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        #d7 = self.down7(d6)
        #d8 = self.down8(d7)

        #u1 = self.up1(d8, d7)
        #u2 = self.up2(u1, d6)
        u3 = self.up3(d6, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

##############################
#        Self-attn block
##############################

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim, return_attention=True):
        super(Self_Attn,self).__init__()
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.return_attention=return_attention
        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        if self.return_attention:
            return out,attention
        else:
            return out

##############################
#        Spectral Norm
##############################

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

#########################################
# Generator with self-attn and spectral norm
#########################################

class GeneratorUNet_Attn(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet_Attn, self).__init__()

        self.noise = GaussianNoise() 

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        #self.attnD1 = Self_Attn(64) # apply at every layer?
        self.down2 = UNetDown(64, 128, spectral_norm=True)
        self.attnD2 = Self_Attn(128) # apply at every layer?
        self.down3 = UNetDown(128, 256, dropout=0.5, spectral_norm=True) 
        self.attnD3 = Self_Attn(256)
        self.down4 = UNetDown(256, 512, dropout=0.5, spectral_norm=True)
        self.attnD4 = Self_Attn(512) 
        #self.attn_down = Self_Attn(512) # 512 -> 512

        self.down5 = UNetDown(512, 512, dropout=0.5, spectral_norm=True)
        self.attnD5 = Self_Attn(512)
        #self.down6 = UNetDown(512, 512, dropout=0.5, spectral_norm=True)
        self.down6 = UNetDown(512, 512, normalize=False, dropout=0.5)
        #self.attnD6 = Self_Attn(512)
        #self.down7 = UNetDown(512, 512, dropout=0.5, spectral_norm=True)
        #self.attnD7 = Self_Attn(512)
        #self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # apply only when input image?
        self.attn = Self_Attn(512)

        #self.up1 = UNetUp(512, 512, dropout=0.5, spectral_norm=True)
        #self.attnU1 = Self_Attn(1024)
        #self.up2 = UNetUp(1024, 512, dropout=0.5, spectral_norm=True)
        #self.attnU2 = Self_Attn(1024)
        #self.up3 = UNetUp(1024, 512, dropout=0.5, spectral_norm=True)
        self.up3 = UNetUp(512, 512, dropout=0.5, spectral_norm=True)
        self.attnU3 = Self_Attn(1024)
        self.up4 = UNetUp(1024, 512, dropout=0.5, spectral_norm=True)
        self.attnU4 = Self_Attn(1024)
        #self.attn_up = Self_Attn(1024) # 1024 as we now have skip connections catted on

        self.up5 = UNetUp(1024, 256, spectral_norm=True)
        self.attnU5 = Self_Attn(256*2)
        self.up6 = UNetUp(512, 128, spectral_norm=True)
        #self.attnU6 = Self_Attn(128*2)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # use noise to help avoid the artifact problem?
        x = self.noise(x)
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        #d1_5, _ = self.attnD1(d1)
        d2 = self.down2(d1)
        d2_5, _ = self.attnD2(d2)
        d3 = self.down3(d2_5)
        d3_5, _ = self.attnD3(d3)
        d4 = self.down4(d3_5)
        #d4_5, _ = self.attn_down(d4) # output attention maps also
        d4_5, _ = self.attnD4(d4)
        d5 = self.down5(d4_5)
        d5_5, _ = self.attnD5(d5)
        d6 = self.down6(d5_5)
        # don't squeeze image own to 1 pixel with 512 channels?
        #d6_5, _ = self.attnD6(d6)
        #d7 = self.down7(d6_5)
        #d7_5, _ = self.attnD7(d7)
        #d8 = self.down8(d7_5)

        #s, _ = self.attn(d8)
        s, _ = self.attn(d6)

        #u1 = self.up1(s, d7_5)
        #u1_5, _ = self.attnU1(u1)
        #u2 = self.up2(u1_5, d6_5)
        #u2_5, _ = self.attnU2(u2)
        #u3 = self.up3(u2_5, d5_5)
        u3 = self.up3(s, d5_5)
        u3_5, _ = self.attnU3(u3)
        u4 = self.up4(u3_5, d4_5)
        u4_5, _ = self.attnU4(u4)
        #u4_5, _ = self.attn_up(u4)
        u5 = self.up5(u4_5, d3_5)
        u5_5, _ = self.attnU5(u5)
        u6 = self.up6(u5_5, d2_5)
        #u6_5, _ = self.attnU6(u6)
        u7 = self.up7(u6, d1)

        return self.final(u7) #, _, _

#####################################
# D is too strong, try adding a bit of noise
# https://www.reddit.com/r/deeplearning/comments/oigdgg/adding_guassian_noise_to_discriminator_layers_in/
# https://github.com/ShivamShrirao/facegan_pytorch/blob/main/facegan_pytorch.ipynb
####################################

class GaussianNoise(nn.Module):
    def __init__(self, std=0.1, decay_rate=0):
        super().__init__()
        self.std = std
        self.decay_rate = decay_rate
        self.training = True

    def decay_step(self):
        self.std = max(self.std - self.decay_rate, 0)

    def forward(self, x):
        if self.training:
            return x + torch.empty_like(x).normal_(std=self.std)
        else:
            return x

#########################################
# Discriminator with self-attn and spectral norm
#########################################

class Discriminator_Attn(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator_Attn, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True, spectral_norm=False, attention=False, std=0.1, std_decay_rate=0):
            """Returns downsampling layers of each discriminator block"""
            self.std = std
            self.std_decay_rate = std_decay_rate

            # try adding some gaussian noise to stop D being too strong (and help G quality)
            layers = [GaussianNoise(self.std, self.std_decay_rate)]

            if spectral_norm:
                #layers = [SpectralNorm(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1))]
                layers.append(SpectralNorm(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)))
            else:
                #layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
                layers.append(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1))
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if attention:
                layers.append(Self_Attn(out_filters, return_attention=False))
            return layers

        self.model0 = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128, spectral_norm=True, attention=True),
            *discriminator_block(128, 256, spectral_norm=True),
            *discriminator_block(256, 512, spectral_norm=True, attention=True)
        )

        #self.attn = Self_Attn(512)

        self.model1 = nn.Sequential(
            GaussianNoise(self.std, self.std_decay_rate), # try gaussian noise here?
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

        # try dropout?

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        out0 = self.model0(img_input)
        #attn, a1 = self.attn(out0)
        out1 = self.model1(out0)
        return out1

########################################
##### Try including some noise similar to
##### Dynamic-Pix2Pix: Noise Injected cGAN for Modeling Input and Target Domain Joint Distributions with Limited Training Data
##### Need a bottle neck module (to stop noise correlations being learnt), and a dynamic switching GAN dependent on whether noise or image is being input
########################################

# The bottleneck is not actually avaible, so this is largely a guess
class UNET_BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        # shrink to 1 x 4 x 4 with conv2d and max pooling (no other info is really given in the paper..., the conv is a guess)
        self.shrink = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)
        # this conv doesn't matter a huge amount as we don't want the encoder to learn anything anyway about the noise
        # It's more about getting noise into the decoder
        self.norm = nn.MaxPool2d(kernel_size=1, stride=1)

    def forward(self, x):
        x = self.shrink(x)
        return self.norm(x)

class GeneratorUNet_BottleNeck(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        """
        add noise to zero regions (representing artifacts) to see if that helps avoid artifact problem
        """
        super(GeneratorUNet_BottleNeck, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5, normalize=False)
        #self.down7 = UNetDown(512, 512, dropout=0.5)
        #self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.bottleneck = UNET_BottleNeck(512, 512) # bottleneck noise (only apply when noise input)
        self.upsample = nn.Upsample(scale_factor=64,  mode='bilinear', align_corners=True) #  multiply noise (4 * 4) to 256 size

        #self.up1 = UNetUp(512, 512, dropout=0.5)
        #self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(512, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, noise_input=False):

        b, c, h, w = x.shape

        # create noise
        noise = torch.rand((b, c, 4, 4)).to(x.device) # (B, C, H, W)
        noise = self.upsample(noise)
        # where x is zero, add upscaled noise
        if noise_input:
            with torch.no_grad():
                d1 = self.down1(x)
                d2 = self.down2(d1)
                d3 = self.down3(d2)
                d4 = self.down4(d3)
                d5 = self.down5(d4)
                d6 = self.down6(d5)
                #d7 = self.down7(d6)
                #d8 = self.down8(d7)
                d6 = self.bottleneck(d6)
        else:
            x[x==-1.] = noise[x==-1.]

            # U-Net generator with skip connections from encoder to decoder
            d1 = self.down1(x)
            d2 = self.down2(d1)
            d3 = self.down3(d2)
            d4 = self.down4(d3)
            d5 = self.down5(d4)
            d6 = self.down6(d5)
            #d7 = self.down7(d6)
            #d8 = self.down8(d7)

        #u1 = self.up1(d8, d7)
        #u2 = self.up2(u1, d6)
        u3 = self.up3(d6, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        if noise_input:
            return self.final(u7), noise
        else:
            return self.final(u7)


#########################################
# Generator with self-attn and spectral norm
#########################################

class GeneratorUNet_Attn_Bottleneck(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet_Attn_Bottleneck, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        #self.attnD1 = Self_Attn(64) # apply at every layer?
        self.down2 = UNetDown(64, 128, spectral_norm=True)
        self.attnD2 = Self_Attn(128) # apply at every layer?
        self.down3 = UNetDown(128, 256, dropout=0.5, spectral_norm=True)
        self.attnD3 = Self_Attn(256)
        self.down4 = UNetDown(256, 512, dropout=0.5, spectral_norm=True)
        self.attnD4 = Self_Attn(512)
        #self.attn_down = Self_Attn(512) # 512 -> 512

        self.down5 = UNetDown(512, 512, dropout=0.5, spectral_norm=True)
        self.attnD5 = Self_Attn(512)
        #self.down6 = UNetDown(512, 512, dropout=0.5, spectral_norm=True)
        self.down6 = UNetDown(512, 512, normalize=False, dropout=0.5)
        #self.attnD6 = Self_Attn(512)
        #self.down7 = UNetDown(512, 512, dropout=0.5, spectral_norm=True)
        #self.attnD7 = Self_Attn(512)
        #self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.bottleneck = UNET_BottleNeck(512, 512) # bottleneck noise (only apply when noise input)
        self.upsample = nn.Upsample(scale_factor=64,  mode='bilinear', align_corners=True) #  multiply noise (4 * 4) to 256 size

        # apply only when input image?
        self.attn = Self_Attn(512)

        #self.up1 = UNetUp(512, 512, dropout=0.5, spectral_norm=True)
        #self.attnU1 = Self_Attn(1024)
        #self.up2 = UNetUp(1024, 512, dropout=0.5, spectral_norm=True)
        #self.attnU2 = Self_Attn(1024)
        #self.up3 = UNetUp(1024, 512, dropout=0.5, spectral_norm=True)
        self.up3 = UNetUp(512, 512, dropout=0.5, spectral_norm=True)
        self.attnU3 = Self_Attn(1024)
        self.up4 = UNetUp(1024, 512, dropout=0.5, spectral_norm=True)
        self.attnU4 = Self_Attn(1024)
        #self.attn_up = Self_Attn(1024) # 1024 as we now have skip connections catted on

        self.up5 = UNetUp(1024, 256, spectral_norm=True)
        self.attnU5 = Self_Attn(256*2)
        self.up6 = UNetUp(512, 128, spectral_norm=True)
        #self.attnU6 = Self_Attn(128*2)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, noise_input=False):
        # use noise to help avoid the artifact problem?
        b, c, h, w = x.shape

        if noise_input:
            # create noise
            noise = torch.rand((b, c, 4, 4)).to(x.device) # (B, C, H, W)
            noise = self.upsample(noise)
            with torch.no_grad():
                # U-Net generator with skip connections from encoder to decoder
                d1 = self.down1(noise)
                #d1_5, _ = self.attnD1(d1)
                d2 = self.down2(d1)
                d2_5, _ = self.attnD2(d2)
                d3 = self.down3(d2_5)
                d3_5, _ = self.attnD3(d3)
                d4 = self.down4(d3_5)
                #d4_5, _ = self.attn_down(d4) # output attention maps also
                d4_5, _ = self.attnD4(d4)
                d5 = self.down5(d4_5)
                d5_5, _ = self.attnD5(d5)
                d6 = self.down6(d5_5)
                # don't squeeze image own to 1 pixel with 512 channels?
                #d6_5, _ = self.attnD6(d6)
                #d7 = self.down7(d6_5)
                #d7_5, _ = self.attnD7(d7)
                #d8 = self.down8(d7_5)
                d6 = self.bottleneck(d6)
        else:
                d1 = self.down1(x)
                #d1_5, _ = self.attnD1(d1)
                d2 = self.down2(d1)
                d2_5, _ = self.attnD2(d2)
                d3 = self.down3(d2_5)
                d3_5, _ = self.attnD3(d3)
                d4 = self.down4(d3_5)
                #d4_5, _ = self.attn_down(d4) # output attention maps also
                d4_5, _ = self.attnD4(d4)
                d5 = self.down5(d4_5)
                d5_5, _ = self.attnD5(d5)
                d6 = self.down6(d5_5)

        #s, _ = self.attn(d8)
        s, _ = self.attn(d6)

        #u1 = self.up1(s, d7_5)
        #u1_5, _ = self.attnU1(u1)
        #u2 = self.up2(u1_5, d6_5)
        #u2_5, _ = self.attnU2(u2)
        #u3 = self.up3(u2_5, d5_5)
        u3 = self.up3(s, d5_5)
        u3_5, _ = self.attnU3(u3)
        u4 = self.up4(u3_5, d4_5)
        u4_5, _ = self.attnU4(u4)
        #u4_5, _ = self.attn_up(u4)
        u5 = self.up5(u4_5, d3_5)
        u5_5, _ = self.attnU5(u5)
        u6 = self.up6(u5_5, d2_5)
        #u6_5, _ = self.attnU6(u6)
        u7 = self.up7(u6, d1)

        if noise_input:
            return self.final(u7), _, _, noise
        else:
            return self.final(u7), _, _

