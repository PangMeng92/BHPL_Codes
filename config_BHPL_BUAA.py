# -*- coding: utf-8 -*-
from easydict import EasyDict as edict

def get_config():
        
    conf = edict()
    conf.batch_size = 5
    conf.lr = 0.0002
    conf.beta1 = 0.5
    conf.beta2 = 0.999
    conf.epochs = 300
    conf.save_dir = './saved_modelBUAA'
    conf.rootA='./BUAA'
    conf.rootB='./BUAA'
    conf.savefig='./BUAA'
    conf.fileA='./dataset/LoadBUAA_nir_trainA.txt'
    conf.fileB='./dataset/LoadBUAA_vis_trainA.txt'
#    conf.fileB='./dataset/LoadBUAA_vis_galleryL.txt'
#    conf.fileA='./dataset/LoadBUAA_testL.txt'
    conf.nd = 50
    conf.nz = 50
    conf.TrainTag=True
    return conf# 
