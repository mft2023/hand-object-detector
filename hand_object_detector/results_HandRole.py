# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 17:20:12 2023

@author: Meng-Fen Tsai
"""

import os
#import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import glob
import torch.optim as optim
from datetime import datetime
import gzip
import numpy as np
import argparse
import linecache

def parse_args():
    parser=argparse.ArgumentParser(description='run hand role classification')
    parser.add_argument('--subj',dest='subj',help='which subject',type=int)
    
    args=parser.parse_args()
    return args
args=parse_args();


class HandRoleDataset(Dataset):
    """ load enture dataset in the GPU memory to speed up training """
    """ get filename, hand_id, handrole feature, label """
    """ When creating dataloader, the __init__ section (in load_handrole_dataset.py) has to be defined."""
    """ In the training, it only requires to have __getitem__. """
                    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):      
        imagename=self.imagename[idx];
        label=self.labels[idx];
        handrole_fea=self.handrole_fea[idx];  
        hand_info=self.hand_info[idx];        
        
        summary=[imagename, hand_info, handrole_fea, label];
        
        return summary
    
###### define new classifier #######
# Define a convolution neural network
class HandRole(nn.Module):
    def __init__(self):
        super(HandRole, self).__init__()
        self.fc1 = nn.Linear(32, 2)

    def forward(self, input_fea):
        output=self.fc1(input_fea)
        
        return output

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using CUDA: ',device)
# Instantiate a neural network model 
handrole = HandRole()

with torch.no_grad():
    handrole.cuda()
    
handrole.to(device)
handrole.train()

print('load HandRole model successfully!')
print('Hand Role architecture: \n',handrole,'\n')

###########################################################
subject=str(args.subj);
num_epochs=100;
dataloader_folder='{path to testing set data loader}/DataLoader_HandRole/Sub'+subject+'/';
save_folder='{path to test set labels}/results/HandRole/LOSOCV_manip_Home/';
save_results_path=save_folder+'/Sub'+subject+'/Home_sub'+subject+'.txt';
save_model_path='{path to best model}/checkpoint_epoch_{epoch#}.pickle';
test_loader_list=glob.glob(dataloader_folder+'/Dataloader/Home_test_loader_*.gz');

if len(save_model_path)>0:# has pretrained model
    print('Loaded model from: '+save_model_path)
    checkpoint=torch.load(save_model_path)
    handrole.load_state_dict(checkpoint['model_state_dict'])
else:
    print('Please provide pretrained model path as save_model_path')

for j in range(len(test_loader_list)):
    test_dataset=torch.load(gzip.GzipFile(test_loader_list[j],'rb'));
    test_loader=DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=0)
    del test_dataset
    len_est_loader=len(test_loader);
    for i, test_batch in enumerate(test_loader):
        framename=test_batch[0];
        hand_id=test_batch[1][0];#hand id: {0:'L', 1:'R'}
        hand_bbx=test_batch[1][1];
        handrole_fea=test_batch[2];
        label=test_batch[3];
        
        handrole_fea.to(device)
        #prediction
        prob = handrole(handrole_fea) 
        prob = prob.reshape(len(label),2);# to match label number (1,num_classes)
        pred=[];
        for k in range(len(label)):
            if prob[k][0]>prob[k][1]:
                pred.append(0)
            else:
                pred.append(1)
        ########## save predictions and labels ###############
        if os.path.exists(save_results_path):
            save_file_path=open(save_results_path,'a')#append filename to .txt            
        else:
            save_file_path=open(save_results_path,'w')#create a new .txt
        save_file_path.write('filename: '+framename+', prediction: '+str(pred)+', labels: '+str(label)+'\n')
        save_file_path.close()
        
    torch.cuda.empty_cache()

    