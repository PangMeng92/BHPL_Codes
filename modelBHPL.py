import torch
import math
from torch import nn, optim
import torch.nn.functional as F


class DiscriminatorA(nn.Module):
    """
    multi-task CNN for predicting ID, and distinguishing real vs. fake prototype in domain A

    ### init
    Nd : Number of identitiy to classify
    """

    def __init__(self, Nd, channel_num):
        super(DiscriminatorA, self).__init__()
        convLayers = [
            nn.Conv2d(channel_num, 32, 3, 1, 1, bias=False), # Bxchx128x128 -> Bx32x128x128
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False), # Bx32x128x128 -> Bx64x128x128
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx64x128x128 -> Bx64x129x129
            nn.Conv2d(64, 64, 3, 2, 0, bias=False), # Bx64x129x128 -> Bx64x64x64
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), # Bx64x64x64 -> Bx64x64x64
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), # Bx64x64x64 -> Bx128x64x64
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx128x64x64 -> Bx128x65x65
            nn.Conv2d(128, 128, 3, 2, 0, bias=False), #  Bx128x65x65 -> Bx128x32x32
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 96, 3, 1, 1, bias=False), #  Bx128x32x32 -> Bx96x32x32
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.Conv2d(96, 192, 3, 1, 1, bias=False), #  Bx96x32x32 -> Bx192x32x32
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx192x32x32 -> Bx192x33x33
            nn.Conv2d(192, 192, 3, 2, 0, bias=False), # Bx192x33x33 -> Bx192x16x16
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.Conv2d(192, 128, 3, 1, 1, bias=False), # Bx192x16x16 -> Bx128x16x16
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), # Bx128x16x16 -> Bx256x16x16
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx256x16x16 -> Bx256x17x17
            nn.Conv2d(256, 256, 3, 2, 0, bias=False),  # Bx256x17x17 -> Bx256x8x8
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 160, 3, 1, 1, bias=False), # Bx256x8x8 -> Bx160x8x8
            nn.BatchNorm2d(160),
            nn.ELU(),
            nn.Conv2d(160, 320, 3, 1, 1, bias=False), # Bx160x8x8 -> Bx320x8x8
            nn.BatchNorm2d(320),
            nn.ELU(),
            nn.AvgPool2d(8, stride=1), #  Bx320x8x8 -> Bx320x1x1
        ]

        self.convLayers = nn.Sequential(*convLayers)
        self.fc = nn.Linear(320, Nd+1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)

    def forward(self, input):
        x = self.convLayers(input)

        x = x.view(-1, 320)

        x = self.fc(x) # Bx320 -> B x (Nd+1)

        return x


class DiscriminatorB(nn.Module):
    """
    multi-task CNN for predicting ID, and distinguishing real vs. fake prototype in domain B

    ### init
    Nd : Number of identitiy to classify
    """

    def __init__(self, Nd, channel_num):
        super(DiscriminatorB, self).__init__()
        convLayers = [
            nn.Conv2d(channel_num, 32, 3, 1, 1, bias=False), # Bxchx128x128 -> Bx32x128x128
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False), # Bx32x128x128 -> Bx64x128x128
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx64x128x128 -> Bx64x129x129
            nn.Conv2d(64, 64, 3, 2, 0, bias=False), # Bx64x129x128 -> Bx64x64x64
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), # Bx64x64x64 -> Bx64x64x64
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), # Bx64x64x64 -> Bx128x64x64
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx128x64x64 -> Bx128x65x65
            nn.Conv2d(128, 128, 3, 2, 0, bias=False), #  Bx128x65x65 -> Bx128x32x32
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 96, 3, 1, 1, bias=False), #  Bx128x32x32 -> Bx96x32x32
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.Conv2d(96, 192, 3, 1, 1, bias=False), #  Bx96x32x32 -> Bx192x32x32
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx192x32x32 -> Bx192x33x33
            nn.Conv2d(192, 192, 3, 2, 0, bias=False), # Bx192x33x33 -> Bx192x16x16
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.Conv2d(192, 128, 3, 1, 1, bias=False), # Bx192x16x16 -> Bx128x16x16
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), # Bx128x16x16 -> Bx256x16x16
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx256x16x16 -> Bx256x17x17
            nn.Conv2d(256, 256, 3, 2, 0, bias=False),  # Bx256x17x17 -> Bx256x8x8
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 160, 3, 1, 1, bias=False), # Bx256x8x8 -> Bx160x8x8
            nn.BatchNorm2d(160),
            nn.ELU(),
            nn.Conv2d(160, 320, 3, 1, 1, bias=False), # Bx160x8x8 -> Bx320x8x8
            nn.BatchNorm2d(320),
            nn.ELU(),
            nn.AvgPool2d(8, stride=1), #  Bx320x8x8 -> Bx320x1x1
        ]

        self.convLayers = nn.Sequential(*convLayers)
        self.fc = nn.Linear(320, Nd+1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)

    def forward(self, input):
        x = self.convLayers(input)

        x = x.view(-1, 320)

        x = self.fc(x) # Bx320 -> B x (Nd+1)

        return x
    

 
    
class Crop(nn.Module):

    def __init__(self, crop_list):
        super(Crop, self).__init__()

        # crop_lsit = [crop_top, crop_bottom, crop_left, crop_right]
        self.crop_list = crop_list

    def forward(self, x):
        B,C,H,W = x.size()
        x = x[:,:, self.crop_list[0] : H - self.crop_list[1] , self.crop_list[2] : W - self.crop_list[3]]

        return x


class Generator(nn.Module):

    def __init__(self, channel_num):
        super(Generator, self).__init__()
        self.features = []
        
        G_prototype_convLayers = [
            nn.Linear(256, 256),
        ]
        
#        G_prototype_convLayers = [
#            nn.Linear(256, 256),
#            nn.Linear(256, 256),
#        ]
        
        
        self.G_prototype_convLayers = nn.Sequential(*G_prototype_convLayers)
    


        GAB_dec_convLayers = [
            nn.ConvTranspose2d(320,160, 3,1,1, bias=False), # Bx320x8x8 -> Bx160x8x8
            nn.BatchNorm2d(160),
            nn.ELU(),
            nn.ConvTranspose2d(160, 256, 3,1,1, bias=False), # Bx160x8x8 -> Bx256x8x8
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ConvTranspose2d(256, 256, 3,2,0, bias=False), # Bx256x8x8 -> Bx256x17x17
            nn.BatchNorm2d(256),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(256, 128, 3,1,1, bias=False), # Bx256x16x16 -> Bx128x16x16
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ConvTranspose2d(128, 192,  3,1,1, bias=False), # Bx128x16x16 -> Bx192x16x16
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.ConvTranspose2d(192, 192,  3,2,0, bias=False), # Bx128x16x16 -> Bx192x33x33
            nn.BatchNorm2d(192),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(192, 96,  3,1,1, bias=False), # Bx192x32x32 -> Bx96x32x32
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.ConvTranspose2d(96, 128,  3,1,1, bias=False), # Bx96x32x32 -> Bx128x32x32
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ConvTranspose2d(128, 128,  3,2,0, bias=False), # Bx128x32x32 -> Bx128x65x65
            nn.BatchNorm2d(128),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(128, 64,  3,1,1, bias=False), # Bx128x64x64 -> Bx64x64x64
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 64,  3,1,1, bias=False), # Bx64x64x64 -> Bx64x64x64
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 64,  3,2,0, bias=False), # Bx64x64x64 -> Bx64x129x129
            nn.BatchNorm2d(64),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(64, 32,  3,1,1, bias=False), # Bx64x128x128 -> Bx32x128x128
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.ConvTranspose2d(32, channel_num,  3,1,1, bias=False), # Bx32x128x128 -> Bxchx128x128
            nn.Tanh(),
        ]
        
        GBA_dec_convLayers = [
            nn.ConvTranspose2d(320,160, 3,1,1, bias=False), # Bx320x8x8 -> Bx160x8x8
            nn.BatchNorm2d(160),
            nn.ELU(),
            nn.ConvTranspose2d(160, 256, 3,1,1, bias=False), # Bx160x8x8 -> Bx256x8x8
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ConvTranspose2d(256, 256, 3,2,0, bias=False), # Bx256x8x8 -> Bx256x17x17
            nn.BatchNorm2d(256),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(256, 128, 3,1,1, bias=False), # Bx256x16x16 -> Bx128x16x16
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ConvTranspose2d(128, 192,  3,1,1, bias=False), # Bx128x16x16 -> Bx192x16x16
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.ConvTranspose2d(192, 192,  3,2,0, bias=False), # Bx128x16x16 -> Bx192x33x33
            nn.BatchNorm2d(192),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(192, 96,  3,1,1, bias=False), # Bx192x32x32 -> Bx96x32x32
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.ConvTranspose2d(96, 128,  3,1,1, bias=False), # Bx96x32x32 -> Bx128x32x32
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ConvTranspose2d(128, 128,  3,2,0, bias=False), # Bx128x32x32 -> Bx128x65x65
            nn.BatchNorm2d(128),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(128, 64,  3,1,1, bias=False), # Bx128x64x64 -> Bx64x64x64
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 64,  3,1,1, bias=False), # Bx64x64x64 -> Bx64x64x64
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 64,  3,2,0, bias=False), # Bx64x64x64 -> Bx64x129x129
            nn.BatchNorm2d(64),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(64, 32,  3,1,1, bias=False), # Bx64x128x128 -> Bx32x128x128
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.ConvTranspose2d(32, channel_num,  3,1,1, bias=False), # Bx32x128x128 -> Bxchx128x128
            nn.Tanh(),
        ]
        
        self.GAB_dec_convLayers = nn.Sequential(*GAB_dec_convLayers)
        self.GBA_dec_convLayers = nn.Sequential(*GBA_dec_convLayers)
        

        self.G_dec_fcAB = nn.Linear(256+50, 320*8*8)
        
        self.G_dec_fcBA = nn.Linear(256+50, 320*8*8)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)



    def forward(self, inputA, inputB, noiseA, noiseB):

        profeaA = self.G_prototype_convLayers(inputA)        
        profeaB = self.G_prototype_convLayers(inputB)

        
        fusefeaA=torch.cat([profeaA, noiseA], 1)
        fusefeaB=torch.cat([profeaB, noiseB], 1)
        
        self.featureA = profeaA
        self.featureB = profeaB
        
                
        feaAB_fc = self.G_dec_fcAB(fusefeaA)
        feaBA_fc = self.G_dec_fcBA(fusefeaB) 

        
        feaAB_fc = feaAB_fc.view(-1, 320, 8, 8)
        feaBA_fc = feaBA_fc.view(-1, 320, 8, 8)
        
        proAB = self.GAB_dec_convLayers(feaAB_fc)
        proBA = self.GBA_dec_convLayers(feaBA_fc)       

        
        return proBA, proAB, profeaA, profeaB


    def forward_a(self, inputA, noiseA):

        profeaA = self.G_prototype_convLayers(inputA)        
        
        self.featureA = profeaA
        
        fusefeaA=torch.cat([profeaA, noiseA], 1)        
        feaAB_fc = self.G_dec_fcAB(fusefeaA)

        
        feaAB_fc = feaAB_fc.view(-1, 320, 8, 8)
        proAB = self.GAB_dec_convLayers(feaAB_fc)
        
        return proAB, profeaA
    
   
    def forward_b(self, inputB, noiseB):

        profeaB = self.G_prototype_convLayers(inputB)        
        
        self.featureB = profeaB
        
        fusefeaB=torch.cat([profeaB, noiseB], 1)         
        feaBA_fc = self.G_dec_fcBA(fusefeaB)

        
        feaBA_fc = feaBA_fc.view(-1, 320, 8, 8)
        proBA = self.GBA_dec_convLayers(feaBA_fc)
        
        return proBA, profeaB
  
    
class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x

class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out

class network_9layers(nn.Module):
    def __init__(self, num_classes=79077):
        super(network_9layers, self).__init__()
        self.features = nn.Sequential(
            mfm(1, 48, 5, 1, 2), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(48, 96, 3, 1, 1), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            )
        self.fc1 = mfm(8*8*128, 256, type=0)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        out = self.fc2(x)
        return out, x

class network_29layers(nn.Module):
    def __init__(self, block, layers, num_classes=79077):
        super(network_29layers, self).__init__()
        self.conv1  = mfm(1, 48, 5, 1, 2)
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = group(48, 96, 3, 1, 1)
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = group(96, 192, 3, 1, 1)
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = group(128, 128, 3, 1, 1)
        self.pool4  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.fc     = mfm(8*8*128, 256, type=0)
        self.fc2    = nn.Linear(256, num_classes)
            
    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.group1(x)
        x = self.pool2(x)

        x = self.block2(x)
        x = self.group2(x)
        x = self.pool3(x)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        fc = F.dropout(fc, training=self.training)
        out = self.fc2(fc)
        return out, fc


class network_29layers_v2(nn.Module):
    def __init__(self, block, layers, num_classes=79077):
        super(network_29layers_v2, self).__init__()
        self.conv1    = mfm(1, 48, 5, 1, 2)
        self.block1   = self._make_layer(block, layers[0], 48, 48)
        self.group1   = group(48, 96, 3, 1, 1)
        self.block2   = self._make_layer(block, layers[1], 96, 96)
        self.group2   = group(96, 192, 3, 1, 1)
        self.block3   = self._make_layer(block, layers[2], 192, 192)
        self.group3   = group(192, 128, 3, 1, 1)
        self.block4   = self._make_layer(block, layers[3], 128, 128)
        self.group4   = group(128, 128, 3, 1, 1)
        self.fc       = nn.Linear(8*8*128, 256)
        self.fc2 = nn.Linear(256, num_classes, bias=False)
            
    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block1(x)
        x = self.group1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block2(x)
        x = self.group2(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        x = F.dropout(fc, training=self.training)
        out = self.fc2(x)
        return out, fc

def LightCNN_9Layers(**kwargs):
    model = network_9layers(**kwargs)
    return model

def LightCNN_29Layers(**kwargs):
    model = network_29layers(resblock, [1, 2, 3, 4], **kwargs)
    return model

def LightCNN_29Layers_v2(**kwargs):
    model = network_29layers_v2(resblock, [1, 2, 3, 4], **kwargs)
    return model