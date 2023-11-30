This repository stores open-source codes for the publication: [Recognizing hand use and hand role at home after stroke from egocentric video](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000361).  

# Hand Role Classification using Hand Object Detector  
In this repository is modified from the original [Hand Object Detector](https://github.com/ddshan/hand_object_detector) to extract hand features for hand role classification. Please clone the original GitHub and replace the _hand-object-detector_ folder with the _hand-object-detector_ folder in this repository. 
## 1. Create three folders
_Datasets_ folder: stores raw images from Home and HomeLab datasets. The structure was _Datasets_/{dataset name}/{participant id}/{video id}/images.  
_LOSOCV_manip_Home_labels_ folder: stores filename, hand side, label {0: stabilization, 1: manipulation} for Home dataset. Labels for the HomeLab dataset is the same format.  
_Results_ folder: stores hand role prediction for each image, including hand side and prediction {0: stabilization, 1: manipulation}.  

## 2. Create data loader to minimize GPU memory usage.
In [load_handrole_dataset.py](hand_object_detector/load_handrole_dataset.py), please set the data augmentation mode (flip images) and correct absolute path to the folders of images and labels in order to extract hand features for data loaders. In each loader (mini-batch data), it contains image filename, hand side, hand features, and its label.

## 3. Launch training
Leave-One-Subject-Out-Cross-Validation (LOSOCV) is to test one single participant with train the model with the data from rest participants. Thereofre, the subj here is the participant that not involved in the training set.  
```
CUDA_VISIBLE_DEVICES=0 python train_hand_role.py --subj=[participant id] --cuda
```
The trained model is saved in the _DataLoader_HandRole_/_Sub{participant id}_/_checkpoints_/ folder.  
The training log (Training_process_Home_sub{participant id}.txt) is saved in the _results_/_HandRole_/_{LOSOCV condition}_/ folder.  

## 4. Launch testing 
