This repository stores open-source codes for the publication: [Recognizing hand use and hand role at home after stroke from egocentric video](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000361).  
This repository is modified from the original [Hand Object Detector](https://github.com/ddshan/hand_object_detector) to extract hand features for hand role classification. Please clone the original GitHub and replace the _hand-object-detector_ folder with the _hand-object-detector_ folder in this repository.  

# ***Hand-Object Interaction Detection***  
Hand Object Detector was applied directly.  
Portable object contact predictions were categorized as hand-object interactions, otherwise, as no interaction.  
In addition to predicted images, predictions in text files are also generated simultaneously in the function `draw_hand_mask` in [vis_hand_obj.py](hand_object_detector/lib/model/utils/viz_hand_obj.py)  
```
CUDA_VISIBLE_DEVICES=0 python demo.py --cuda --checkepoch=8 --checkpoint=132028 --subj={participant ID} --vis --image_dir={path to the folder of raw images} --save_dir={path to results folder}
```  
# ***Hand Role Classification***  
## 1. Create three folders
`Datasets` folder: stores raw images from Home and HomeLab datasets. The structure was _Datasets_/{dataset name}/{participant id}/{video id}/images.  
`LOSOCV_manip_Home_labels` folder: stores filename, hand side, label {0: stabilization, 1: manipulation} for Home dataset. Labels for the HomeLab dataset in the same format.  
`Results` folder: stores hand role prediction for each image, including hand side and prediction {0: stabilization, 1: manipulation}.  

## 2. Create data loader to minimize GPU memory usage.
In [load_handrole_dataset.py](hand_object_detector/load_handrole_dataset.py), please set the data augmentation mode (flip images) and the absolute path to the folders of images and labels in order to extract hand features for data loaders. Each loader (mini-batch data) contains image filenames, hand sides, hand features, and labels.

## 3. Launch training
Leave-One-Subject-Out-Cross-Validation (LOSOCV) is to test one single participant by training the model with the data from the rest of the participants. Therefore, the subj here is the participant that not involved in the training set.  
```
CUDA_VISIBLE_DEVICES=0 python train_hand_role.py --cuda --subj=participant_id
```
The trained model is saved in the _DataLoader_HandRole_/_Sub{participant id}_/_checkpoints_/ folder.  
The training log, `Training_process_{LOSOCV_condition}_sub{participant id}.txt`, is saved in the _results_/_HandRole_/_{LOSOCV condition}_/ folder.  

## 4. Launch testing  
Choosing the best model in the validation set in each LOSOCV iteration and testing the model in the data of a left-out participant.  
A tested model and save folder can be determined in [results_HandRole.py](hand_object_detector/results_HandRole.py).
```
CUDA_VISIBLE_DEVICES=0 python results_HandRole.py --cuda --subj=participant_id
```
# Cite
If you find this repository useful in your research, please consider citing:
```
@article{
    Author = {Meng-Fen Tsai,Rosalie H. Wang, and Zariffa, José},
    Title = {Recognizing hand use and hand role at home after stroke from egocentric video},
    Journal = {PLOS Digital Health 2.10 (2023): e0000361},
    Year = {2023}
}
```
