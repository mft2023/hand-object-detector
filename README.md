This repository stores open-source codes for the publication: [Recognizing hand use and hand role at home after stroke from egocentric video](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000361).  

# ***Hand Role Classification*** using Hand Object Detector  
In this repository is modified from the original [Hand Object Detector](https://github.com/ddshan/hand_object_detector) to extract hand features for hand role classification. Please clone the original GitHub and replace the _hand-object-detector_ folder with the _hand-object-detector_ folder in this repository. 
## 1. Create three folders
`Datasets` folder: stores raw images from Home and HomeLab datasets. The structure was _Datasets_/{dataset name}/{participant id}/{video id}/images.  
`LOSOCV_manip_Home_labels` folder: stores filename, hand side, label {0: stabilization, 1: manipulation} for Home dataset. Labels for the HomeLab dataset is the same format.  
`Results` folder: stores hand role prediction for each image, including hand side and prediction {0: stabilization, 1: manipulation}.  

## 2. Create data loader to minimize GPU memory usage.
In [load_handrole_dataset.py](hand_object_detector/load_handrole_dataset.py), please set the data augmentation mode (flip images) and correct absolute path to the folders of images and labels in order to extract hand features for data loaders. In each loader (mini-batch data), it contains image filename, hand side, hand features, and its label.

## 3. Launch training
Leave-One-Subject-Out-Cross-Validation (LOSOCV) is to test one single participant with training the model with the data from the rest of participants. Thereofre, the subj here is the participant that not involved in the training set.  
```
CUDA_VISIBLE_DEVICES=0 python train_hand_role.py --cuda --subj=participant_id
```
The trained model is saved in the _DataLoader_HandRole_/_Sub{participant id}_/_checkpoints_/ folder.  
The training log, `Training_process_{LOSOCV_condition}_sub{participant id}.txt`, is saved in the _results_/_HandRole_/_{LOSOCV condition}_/ folder.  

## 4. Launch testing  
Choosing the best model in the validation set in each LOSOCV iteration and test it in the left out participant.  
I also generated predictions in text files while generating image output in function `draw_hand_mask` in [vis_hand_obj.py](hand_object_detector/lib/model/utils/viz_hand_obj.py)  

```
CUDA_VISIBLE_DEVICES=0 python results_HandRole.py --cuda --subj=participant_id
```

