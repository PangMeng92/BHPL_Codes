# -*- coding: utf-8 -*-
from modelBHPL import DiscriminatorA, DiscriminatorB, Generator, LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2
#from readerDisguise import get_batch
#from readerLight import get_batch
#from readerPV import get_batch
#from readerPoseFEI import get_batch
#from readerFuse import get_batch
from readerData import get_batchA, get_batchB
import torch as t
from torch import nn, optim
from config_BHPL_BUAA import get_config
import numpy as np
from tensorboardX import SummaryWriter
from torchvision import transforms
import torchvision.utils as vutils
import cv2 as cv
import visdom
import os
import torch.nn.functional as F
from utils import *
from scipy import misc
import imageio

def one_hot(label,depth):
    ones = t.sparse.torch.eye(depth)
    return ones.index_select(0,label)


def generateBHPL(conf):
    vis = visdom.Visdom()
    G = Generator(1).cuda()
    
    T=t.load('./saved_modelBUAA/BUAAH%2d.pth'%280)   # A270 WODgan250
        
    G.load_state_dict(T['g_net_list'])
    G.eval()
    
    Extractor = LightCNN_29Layers_v2(num_classes=150)
    Extractor.eval()
    Extractor = nn.DataParallel(Extractor).cuda()
    
    checkpoint=t.load('./lightCNN_BUAA_vis_nir_epoch_40_iter_0.pth')
    Extractor.load_state_dict(checkpoint['model'])
        
    train_loaderA = get_batchA(conf.rootA, conf.fileA, conf.batch_size)    #x
    train_loaderB = get_batchB(conf.rootB, conf.fileB, conf.batch_size)

    resA = []
    id_labA = []
    
    resB = []
    id_labB = []
    
    for epoch in range(1,conf.epochs+1):
        print('%d epoch ...'%(epoch))
        for i, (loaderA, loaderB) in enumerate(zip(train_loaderA, train_loaderB)):
            batchA_data = loaderA
            batchB_data = loaderB
            
            batchA_image = batchA_data[0]
            batchA_id_label = batchA_data[1]
            batchA_var_label = batchA_data[2]
            batchA_pro = batchA_data[3]
            batchA_proB = batchA_data[4]
            
            batchB_image = batchB_data[0]
            batchB_id_label = batchB_data[1]
            batchB_var_label = batchB_data[2]
            batchB_pro = batchB_data[3]
            batchB_proA = batchB_data[4]
            
            fixed_noiseA = t.FloatTensor(np.random.uniform(-1, 1, (conf.batch_size, conf.nz)))
            fixed_noiseB = t.FloatTensor(np.random.uniform(-1, 1, (conf.batch_size, conf.nz)))  
            
            #cuda
            batchA_image, batchA_id_label, batchA_var_label, batchA_pro, batchA_proB = \
                batchA_image.cuda(), batchA_id_label.cuda(), batchA_var_label.cuda(), batchA_pro.cuda(), batchA_proB.cuda()
                
            batchB_image, batchB_id_label, batchB_var_label, batchB_pro, batchB_proA = \
                batchB_image.cuda(), batchB_id_label.cuda(), batchB_var_label.cuda(), batchB_pro.cuda(), batchB_proA.cuda()     
                        
            fixed_noiseA, fixed_noiseB = fixed_noiseA.cuda(), fixed_noiseB.cuda() 
            
            _, FeaA = Extractor(batchA_image)
            _, FeaB = Extractor(batchB_image)
            
            proAB, profeaA = G.forward_a(FeaA, fixed_noiseA)
            proBA, profeaB = G.forward_b(FeaB, fixed_noiseB)
            
            
            if i % 1 == 0:
                # x = vutils.make_grid(generated, normalize=True, scale_each=True)
                # writer.add_image('Image', x, i)
                batchA_image =  batchA_image.cpu().data.numpy()/2+0.5
                proAB = proAB.cpu().data.numpy()/2+0.5
                batchA_proB = batchA_proB.cpu().data.numpy()/2+0.5
                
                vis.images(batchA_image,nrow=2,win='batchA_image',opts=dict(caption='batchA_image'))
                vis.images(proAB,nrow=2,win='proAB',opts=dict(caption='proAB'))
                vis.images(batchA_proB,nrow=2,win='batchA_proB',opts=dict(caption='batchA_proB'))
                
                batchB_image = batchB_image.cpu().data.numpy()/2+0.5
                proBA = proBA.cpu().data.numpy()/2+0.5
                batchB_proA = batchB_proA.cpu().data.numpy()/2+0.5
                
                vis.images(batchB_image,nrow=2,win='batchB_image',opts=dict(caption='batchB_image'))
                vis.images(proBA,nrow=2,win='proBA',opts=dict(caption='proBA'))
                vis.images(batchB_proA,nrow=2,win='batchB_proA',opts=dict(caption='batchB_proA'))
                
               

                
if __name__=='__main__':
    conf = get_config()
    print(conf)
    generateBHPL(conf) 
    

