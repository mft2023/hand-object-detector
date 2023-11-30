#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 20:48:28 2022

@author: tsaim
"""
import os
import linecache
from sklearn.metrics import f1_score, matthews_corroef

filepath='{path to results folder}/HandRole_Shan/LOSOCV_manip_Home_HomeLab/Sub2/LOSOCV_manip_Home_HomeLab_sub2.txt';
labeling_file=open(filepath, "r")
num_instances=len(labeling_file.readlines());
labeling_file.close()
    
for i in range(1,num_instances):#line 0 is nothing
    current_image=linecache.getline(filepath,i).split(' - ')[0];
    #print(current_image)
    dset=current_image[0:2];
    subj=current_image[2:4];
    vset=current_image[6:8];