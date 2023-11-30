#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 20:31:43 2022

@author: tsaim
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
    parser.add_argument('--subj',dest='subj',help='which leave out subject',type=int)
    
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

def Cal_weight(txt_path):
    target_labels=open(txt_path,"r")
    num_instances=len(target_labels.readlines());
    target_labels.close();
    num_0=0;num_1=0;
    for i in range(1,num_instances):
        trained_image=linecache.getline(txt_path,i);
        label=int(trained_image.split(' - ')[1].split(': ')[1][0]);
        if label==1:
            num_1=num_1+1;
        else:
            num_1=num_0+1;
    weights=torch.FloatTensor([1-num_0/(num_instances-1),1-num_1/(num_instances-1)]);
    weights=1/weights;
    
    return weights

# decide loss function and optimizer
optimizer = optim.SGD(handrole.parameters(), lr=0.0001, momentum=0.9, nesterov=True)
lr_scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.001,patience=2,verbose=True)

subject=str(args.subj);
num_epochs=100;
dataloader_folder='{path to your training dataloader folder}/DataLoader_HandRole/Sub'+subject+'/';
dataloader_folder_val='{path to your validation dataloader folder}/DataLoader_HandRole/Sub'+subject+'/';
train_txt_path='{path to the env}/hand_object_detector/LOSOCV_manip_Home_labels/LOSOCV_Home_sub'+subject+'_Manipulation_train.txt';
save_folder='{path to results folder}/HandRole/LOSOCV_manip_Home/';
training_process_path=save_folder+'/Sub'+subject+'/Training_process_Home_sub'+subject+'.txt';
save_model_path='';
train_loader_list=glob.glob(dataloader_folder+'/Dataloader/Home_train_loader_*.gz');
val_loader_list=glob.glob(dataloader_folder_val+'/Dataloader/Home_val_loader_*.gz');

with torch.no_grad():
    weights=torch.FloatTensor([1,20])
fn_loss = nn.CrossEntropyLoss(weight=weights.to(device))

if not os.path.exists(os.path.join(dataloader_folder,'checkpoints')):
    os.makedirs(os.path.join(dataloader_folder,'checkpoints'));
if not os.path.exists(os.path.join(save_folder,'Sub'+subject)):
    os.makedirs(os.path.join(save_folder,'Sub'+subject));
    
start_epoch=0;
if len(save_model_path)>0:# has pretrained model
    print('Loaded model from: '+save_model_path)
    checkpoint=torch.load(save_model_path)
    handrole.load_state_dict(checkpoint['model_state_dict'])
    start_epoch=checkpoint['epoch']
    loss=checkpoint['loss']
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

print('num_split - train_loader: ',len(train_loader_list))
print('num_split - val_loader: ',len(val_loader_list))

########### training ###########
for epoch in range(start_epoch,num_epochs):
    train_loss=0;
    start_loading=datetime.now();
    accuracy_epoch=[];
    for j in range(len(train_loader_list)):
        accuracy=0;
        train_loading1=datetime.now();
        train_dataset=torch.load(gzip.GzipFile(train_loader_list[j],'rb'));
        train_loader=DataLoader(train_dataset, batch_size=20, shuffle=False, num_workers=0)
        train_loading2=datetime.now();
        del train_dataset
        diff_time=(train_loading2-train_loading1).total_seconds();
        len_train_loader=len(train_loader);
        print('Load trian_loader '+str(j)+' takes: %.2f minutes' % (diff_time/60))
        for i, train_batch in enumerate(train_loader):
            framename=train_batch[0];
            hand_id=train_batch[1][0];#{0:'L', 1:'R'}
            hand_bbx=train_batch[1][1];
            handrole_fea=train_batch[2];
            label=train_batch[3];
            
            handrole_fea.to(device)
            label.to(device)            
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            prob = handrole(handrole_fea) 
            prob = prob.reshape(len(label),2);
            pred=[];
            for k in range(len(label)):
                if prob[k][0]>prob[k][1]:
                    pred.append(0)
                else:
                    pred.append(1)
                    
            label = label.reshape(len(label));
            loss = fn_loss(prob.cuda(), label.cuda())
            train_loss =+ loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            accuracy += (len(np.where(pred==label.tolist())[0])/len(label));
            torch.cuda.empty_cache() 
        #delete train_loader
        torch.cuda.empty_cache()
        accuracy_epoch.append(accuracy/len_train_loader)
    lr_scheduler.step(train_loss)
    end_loading=datetime.now();
    diff_time=(end_loading-start_loading).total_seconds();
    print('This training epoch takes: %.2f minutes' % (diff_time/60))
    print('Epoch [{}/{}], Loss: {:.4f}, accuracy: {:4f}'.format(epoch+1, num_epochs, train_loss, sum(accuracy_epoch)))
    ########## save training loss ###############
    if os.path.exists(training_process_path):
        save_file_path=open(training_process_path,'a')           
    else:
        save_file_path=open(training_process_path,'w')
    save_file_path.write('Epoch: '+str(epoch+1)+', train_loss: '+str(train_loss)+', train_accuracy: '+str(sum(accuracy_epoch))+'\n')
    save_file_path.close()
    ############### Validation ##################
    if  ((epoch+1) % 5)==0:
        val_loss_epoch=0;
        num_instance_val=0;
        accuracy_epoch_val=[];
        start_loading=datetime.now();
        for j in range(len(val_loader_list)):
            accuracy_val=0;
            val_dataset=torch.load(gzip.GzipFile(val_loader_list[j],'rb'));
            val_loader=DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=0);
            del val_dataset
            len_val_loader=len(val_loader);
            for i, val_batch in enumerate(val_loader):
                framename_val=val_batch[0];
                hand_id_val=val_batch[1][0];#{0:'L', 1:'R'}
                hand_bbx_val=val_batch[1][1];
                handrole_fea_val=val_batch[2];
                label_val=val_batch[3];                
                    
                num_instance_val=num_instance_val+1;
                handrole_fea_val.to(device)
                label_val.to(device)
                # forward + backward + optimize
                prob_val = handrole(handrole_fea_val);
                prob_val = prob_val.reshape(len(label_val),2);
                
                pred_val=[];
                for k in range(len(label_val)):
                    if prob_val[k][0]>prob_val[k][1]:
                        pred_val.append(0)
                    else:
                        pred_val.append(1)   
                        
                label_val = label_val.reshape(len(label_val));
                val_loss = fn_loss(prob_val.cuda(), label_val.cuda())
                accuracy_val += (len(np.where(pred_val==label_val.tolist())[0])/len(label_val));
                val_loss_epoch=+val_loss.item();
            #delete val_loader
            torch.cuda.empty_cache()
            accuracy_epoch_val.append(accuracy_val/len_val_loader)
        end_loading=datetime.now();
        diff_time=(end_loading-start_loading).total_seconds();
        print('This validation epoch takes: %.2f minutes' % (diff_time/60))
        save_model_path=dataloader_folder+'/checkpoints/checkpoint_epoch_'+str(epoch+1)+'.pickle';
        torch.save({'epoch':epoch+1,'model_state_dict':handrole.state_dict(),'optimizer':optimizer.state_dict(),'loss':loss,'val_loss':val_loss_epoch,'lr_scheduler':lr_scheduler.state_dict()}, save_model_path)
        print('Val Loss: {:.4f}, accuracy: {:4f}'.format(val_loss_epoch,sum(accuracy_epoch_val)))
        ########## save validation loss ###############
        if os.path.exists(training_process_path):
            save_file_path=open(training_process_path,'a')           
        else:
            save_file_path=open(training_process_path,'w')#
        save_file_path.write('Epoch: '+str(epoch+1)+', val_loss: '+str(val_loss_epoch)+', val_accuracy: '+str(sum(accuracy_epoch_val))+'\n')
        save_file_path.close()
        
