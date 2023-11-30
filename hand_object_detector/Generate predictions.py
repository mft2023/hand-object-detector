#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:24:33 2021

@author: tsaim
"""
import os

GT_labeling_folder='';
txt_folder_path='{path to results folder}/hand_object_detector/results/';
files=next(os.walk(txt_folder_path))[2];
filename_hand_pred=[];

# read each filename.txt to get its hand role prediction
for i in range(0,len(files)):
    with open(txt_folder_path+'/'+files[i],'r') as b:
        lines=[line.strip() for line in b];        
    
    for j in range (0,len(lines)):
        if lines[j].split(': ')[1]=='R-N':#non-interaction
            hand='R';
            pred=0;
        elif lines[j].split(': ')[1]=='L-N':#non-interaction
            hand='L';
            pred=0;
        elif lines[j].split(': ')[1]=='R-P':#portable object contact
            hand='R';
            pred=1;
        elif lines[j].split(': ')[1]=='L-P':#portable object contact
            hand='L';
            pred=1;
        elif lines[j].split(': ')[1]=='R-F':#stationary object contact
            hand='R';
            pred=1;
        elif lines[j].split(': ')[1]=='L-F':#stationary object contact
            hand='L';
            pred=1;
        filename_hand_pred.append([files[i],hand,pred]);
            
