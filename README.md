# Hand Role Classification using Hand Object Detector 
This repository stores open-source codes for the publication: [Recognizing hand use and hand role at home after stroke from egocentric video](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000361).  

# 3. Hand Object Detector  
In this repository is modified from the original [Hand Object Detector](https://github.com/ddshan/hand_object_detector) to extract hand features for hand role classification. Please clone the original GitHub and replace the _hand-object-detector_ folder with the _hand-object-detector_ folder in this repository. 
## 1. Create three folders
_Datasets_ folder: stores raw images from Home and HomeLab datasets. The structure was _Datasets_/{dataset name}/{participant id}/{video id}/images.
_LOSOCV_manip_Home_labels_ folder: stores filename, hand side, label {0: stabilization, 1: manipulation} for Home dataset. Labels for the HomeLab dataset is the same format.
_Results_ folder: stores hand role prediction for each image, including hand side and prediction {0: stabilization, 1: manipulation}.
