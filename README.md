# BHPL_Codes

Bidirectional heterogeneous prototype learning (BHPL) model

This work entitled "A Unified Framework for Bidirectional Prototype Learning from Contaminated Faces across Heterogeneous Domains" has been submitted to IEEE Transactions on Information Forensics and Security (TIFS). 
In this package, we implement our DisP+V using Pytorch, and train/test the DisP+V model on BUAA NIR-VIS heterogeneous face dataset.

-------------------------------------------------------------------------
## Train BHPL model:

### Step 1. Open config_BHPL_BUAA.py 

set con.batch_size =5;

set conf.lr = 0.0002;

set conf.beta1 = 0.5;

set conf.beta2 = 0.999;

set conf.epochs = 300;

set conf.fileA='./dataset/LoadBUAA_nir_trainA.txt';

set conf.fileB='./dataset/LoadBUAA_vis_trainA.txt';

conf.nz = 50;

set conf.nd=50;

set conf.TrainTag = True;

### Step 2. Open readerData.py

set shuffle=True in def get_batchA

set shuffle=True in def get_batchB

### Step 3. Run TrainBHPL.py


--------------------------------------------------------------------------
## Generate Heterogeneous Prototype:

### Step 1. Open config_BHPL_BUAA.py 

set con.batch_size =1;

set conf.epochs = 1;

set conf.fileB='./dataset/LoadBUAA_vis_galleryL.txt'

set conf.fileA='./dataset/LoadBUAA_testL.txt'

set conf.TrainTag = False;

### Step 2. Open readerData.py

set shuffle=False in def get_batchA

set shuffle=False in def get_batchB

### Step 3. Run GenerateBHPL.py

choose a trained model, and load it in def generateBHPL


The software is free for academic use, and shall not be used, rewritten, or adapted as the basis of a commercial product without first obtaining permission from the authors. The authors make no representations about the suitability of this software for any purpose. It is provided "as is" without express or implied warranty.
