# -*- coding: utf-8 -*-
from modelBHPL import DiscriminatorA, DiscriminatorB, Generator, LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2
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

def MMD_Loss(fc_nir, fc_vis):
    mean_fc_nir = t.mean(fc_nir, 0)
    mean_fc_vis = t.mean(fc_vis, 0)
    loss_mmd = F.mse_loss(mean_fc_nir, mean_fc_vis)
    return loss_mmd

def trainBHPL(conf):
    vis = visdom.Visdom()
    train_loaderA = get_batchA(conf.rootA, conf.fileA, conf.batch_size)    #x
    train_loaderB = get_batchB(conf.rootB, conf.fileB, conf.batch_size)
    
    DA = DiscriminatorA(conf.nd, 1).cuda()
    DB = DiscriminatorB(conf.nd, 1).cuda()
    
    G = Generator(1).cuda()
    
    
    Extractor = LightCNN_29Layers_v2(num_classes=150)
    Extractor.eval()
    Extractor = nn.DataParallel(Extractor).cuda()
    
    checkpoint=t.load('./lightCNN_BUAA_vis_nir_epoch_40_iter_0.pth')
    Extractor.load_state_dict(checkpoint['model'])
    
    G.train()
    DA.train()
    DB.train()

    optimizer_DA = optim.Adam(DA.parameters(),
                             lr=conf.lr,betas=(conf.beta1,conf.beta2))
    optimizer_DB = optim.Adam(DB.parameters(),
                             lr=conf.lr,betas=(conf.beta1,conf.beta2))        
    optimizer_G = optim.Adam(G.parameters(), lr=conf.lr,
                                           betas=(conf.beta1, conf.beta2))
    
    # Learning rate update schedulers
    lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(conf.epochs, 0, 1).step)
    lr_scheduler_DA = optim.lr_scheduler.LambdaLR(optimizer_DA, lr_lambda=LambdaLR(conf.epochs, 0, 1).step)
    lr_scheduler_DB = optim.lr_scheduler.LambdaLR(optimizer_DB, lr_lambda=LambdaLR(conf.epochs, 0, 1).step)
    
    loss_criterion = nn.CrossEntropyLoss()
    loss_criterion_gan = nn.BCEWithLogitsLoss()

    steps = 0
    # writer = SummaryWriter()

    for epoch in range(0,conf.epochs):
        print('%d epoch ...'%(epoch+1))
        G_loss = 0
        for i, (loaderA, loaderB) in enumerate(zip(train_loaderA, train_loaderB)):
            batchA_data= loaderA
            batchB_data= loaderB
            
            DA.zero_grad()
            DB.zero_grad()
            G.zero_grad()
            
            batchA_image = batchA_data[0]
            batchA_id_label = batchA_data[1]-1
            batchA_var_label = batchA_data[2]
            batchA_pro = batchA_data[3]
            batchA_proB = batchA_data[4]
            
            for j in range(conf.batch_size):
                if batchA_var_label[j]==0:
                    batchA_pro[j]=batchA_image[j]
            
            batchB_image = batchB_data[0]
            batchB_id_label = batchB_data[1]-1
            batchB_var_label = batchB_data[2]
            batchB_pro = batchB_data[3]
            batchB_proA = batchB_data[4]
            
            for j in range(conf.batch_size):
                if batchB_var_label[j]==0:
                    batchB_pro[j]=batchB_image[j]
                    
            batch_ones_label = t.ones(conf.batch_size)  
            batch_zeros_label = t.zeros(conf.batch_size)
            
            fixed_noiseA = t.FloatTensor(np.random.uniform(-1, 1, (conf.batch_size, conf.nz)))
            fixed_noiseB = t.FloatTensor(np.random.uniform(-1, 1, (conf.batch_size, conf.nz)))            

            
            #cuda
            batchA_image, batchA_id_label, batchA_var_label, batchA_pro, batchA_proB = \
                batchA_image.cuda(), batchA_id_label.cuda(), batchA_var_label.cuda(), batchA_pro.cuda(), batchA_proB.cuda()
                
            batchB_image, batchB_id_label, batchB_var_label, batchB_pro, batchB_proA = \
                batchB_image.cuda(), batchB_id_label.cuda(), batchB_var_label.cuda(), batchB_pro.cuda(), batchB_proA.cuda()     
            
            fixed_noiseA, fixed_noiseB = fixed_noiseA.cuda(), fixed_noiseB.cuda() 
            batch_ones_label, batch_zeros_label = batch_ones_label.cuda(), batch_zeros_label.cuda()
            
            _, FeaA = Extractor(batchA_image)
            _, FeaB = Extractor(batchB_image)
            
            proAB, profeaA = G.forward_a(FeaA, fixed_noiseA)
            proBA, profeaB = G.forward_b(FeaB, fixed_noiseB)
            
            
            
            steps += 1
            
            if i%2==0:
                # DiscriminatorA 
                DA_loss = Learn_DA(DA, loss_criterion, loss_criterion_gan, optimizer_DA, batchB_image, batchB_pro, proAB,\
                                            batchB_id_label, batch_ones_label, batch_zeros_label, epoch, steps, conf.nd, conf)
                
                DB_loss = Learn_DB(DB, loss_criterion, loss_criterion_gan, optimizer_DB, batchA_image, batchA_pro, proBA,\
                                            batchA_id_label, batch_ones_label, batch_zeros_label, epoch, steps, conf.nd, conf)
                
            
            else:
                # Generator
                G_loss = Learn_G(G, DA, DB, loss_criterion, loss_criterion_gan, optimizer_G, proAB, proBA, profeaA, profeaB,\
                            batchA_id_label, batchB_id_label, batch_ones_label, batch_zeros_label, epoch, steps, conf.nd, conf)

            if i % 10 == 0:
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
                

                print('%d steps DA loss is %f, DB loss is %f, G loss is %f'%(steps, DA_loss, DB_loss, G_loss))
        
        lr_scheduler_G.step()
        lr_scheduler_DA.step()
        lr_scheduler_DB.step()
        
        if epoch%10 ==0:
            msg = 'Saving checkpoint :{}'.format(epoch)    #restore from epoch+1
            print(msg)
            G_state_list = G.state_dict()
            DA_state_list = DA.state_dict()
            DB_state_list = DB.state_dict()
            t.save({
                'epoch':epoch,
                'g_net_list':G_state_list,
                'dA_net_list' :DA_state_list,
                'dB_net_list' :DB_state_list,
            },
            os.path.join(conf.save_dir,'BUAA%04d.pth'% epoch))

    # writer.close()

 
def Learn_DA(DA, loss_criterion, loss_criterion_gan, optimizer_DA, batchB_image, batchB_pro, proAB,\
            batchB_id_label, batch_ones_label, batch_zeros_label, epoch, steps, Nd, args):

    real_output = DA(batchB_image)
    pro_output = DA(batchB_pro)
    syn_outputAB = DA(proAB.detach()) # .detach()
    
    L_id    = loss_criterion(real_output[:, :Nd], batchB_id_label)
    L_gan   = loss_criterion_gan(pro_output[:, Nd], batch_ones_label) + loss_criterion_gan(syn_outputAB[:, Nd], batch_zeros_label)
    
    
    DA_loss = L_gan + 2*L_id    # lighting 1,5,  pose  1,5 （1，10）, Fuse 1,5

    DA_loss.backward()
    optimizer_DA.step()

    # Discriminator A
    a = DA_loss.cpu().data.item()
    
    return a


def Learn_DB(DB, loss_criterion, loss_criterion_gan, optimizer_DB, batchA_image, batchA_pro, proBA, \
            batchA_id_label, batch_ones_label, batch_zeros_label, epoch, steps, Nd, args):

    real_output = DB(batchA_image)
    proA_output = DB(batchA_pro)
    syn_outputBA = DB(proBA.detach()) # .detach()
    
    L_id    = loss_criterion(real_output[:, :Nd], batchA_id_label)
    L_gan   = loss_criterion_gan(proA_output[:, Nd], batch_ones_label) + loss_criterion_gan(syn_outputBA[:, Nd], batch_zeros_label) 
    
    
    DB_loss = L_gan + 2*L_id    # lighting 1,5,  pose  1,5 （1，10）, Fuse 1,5

    DB_loss.backward()
    optimizer_DB.step()

    # Discriminator B
    b = DB_loss.cpu().data.item()
    
    return b



def Learn_G(G, DA, DB, loss_criterion, loss_criterion_gan, optimizer_G, proAB, proBA, profeaA, profeaB,\
                            batchA_id_label, batchB_id_label, batch_ones_label, batch_zeros_label, epoch, steps, Nd, args):

    
    pro_outputAB = DA(proAB)
    pro_outputBA = DB(proBA)
    
    
    LA_id    = loss_criterion(pro_outputAB[:, :Nd], batchA_id_label)
    LB_id    = loss_criterion(pro_outputBA[:, :Nd], batchB_id_label)
    L_gan    = loss_criterion_gan(pro_outputAB[:, Nd], batch_ones_label) + loss_criterion_gan(pro_outputBA[:, Nd], batch_ones_label)
    
    
    L_pair = 0
    Sim_vector=batchA_id_label-batchB_id_label    
    Index=(Sim_vector==0).nonzero().squeeze()
    L_pair = (profeaA[Index] - profeaB[Index]).pow(2).sum()/args.batch_size
    
    L_mmd = MMD_Loss(profeaA, profeaB)
    
    G_loss = 1*L_gan + 2*(LA_id+LB_id) + 0.1*L_mmd + 0.1*L_pair
    
    G_loss.backward()
    optimizer_G.step()
    
    # Generator
    g = G_loss.cpu().data.item()
    
    return g

                
if __name__=='__main__':
    conf = get_config()
    print(conf)
    trainBHPL(conf)

    

