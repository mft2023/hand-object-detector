#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 13:11:32 2022

@author: tsaim
"""
import os
import numpy as np
import torch
import gzip
from torch.utils.data import Dataset, DataLoader
from model.faster_rcnn.resnet import resnet
import argparse
from model.utils.config import cfg
import cv2
from model.roi_layers import nms
from model.utils.blob import im_list_to_blob
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
import linecache
from datetime import datetime

class HandRoleDataset(Dataset):
    """ load enture dataset in the GPU memory to speed up training """
    """ get filename, hand_id, handrole feature, label """
    def __init__(self, image_path, labeling_path,flip_mode):
        super(HandRoleDataset, self).__init__()
        self.image_path=image_path;
        self.labeling_path=labeling_path;   
        self.imagename=[];
        self.hand_info=[];       
        self.handrole_fea=[];
        self.labels=[]; 
        self.flip_mode=flip_mode
        for i in range(len(self.image_path)):
            hand_info_frame=[];
            img_name=self.image_path[i].split('/')[-1];
            with torch.no_grad():
                ####### get features ###########
                if self.flip_mode=='flip':
                    blobs, im_scales = _get_image_blob(cv2.flip(cv2.imread(self.image_path[i]),1))
                else:
                    blobs, im_scales = _get_image_blob(cv2.imread(self.image_path[i]))
                assert len(im_scales) == 1, "Only single-image batch implemented"
                im_blob = blobs
                im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
                im_data_pt = torch.from_numpy(im_blob)
                im_data_pt = im_data_pt.permute(0, 3, 1, 2)
                im_info_pt = torch.from_numpy(im_info_np)
                
                im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
                im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
                gt_boxes.resize_(1, 1, 5).zero_()
                num_boxes.resize_(1).zero_()
                box_info.resize_(1, 1, 5).zero_() 
                #get feature
                input_value=[]
                features = []
                def save_features(mod,inp,oupt):
                    input_value.append(inp)
                    features.append(oupt)
                # add a hook to extract features
                layer_to_hook =['extension_layer.hand_contact_state_layer.3','extension_layer.hand_lr_layer']#'extension_layer.hand_contact_state_layer.0'
                for name, layer in fasterRCNN.named_modules():
                    if name in layer_to_hook:
                        #1st one is hand state, 2nd one is hand_side
                        layer.register_forward_hook(save_features)
                        
                rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info)
                
                handrole_features=input_value[0][0][0];
                contact_vector=loss_list[0][0]#{0:'No Contact', 1:'Self Contact', 2:'Another Person', 3:'Portable Object', 4:'Stationary Object'}
                offset_vector = loss_list[1][0].detach()#['__background__', 'targetobject', 'hand']
                lr_vector=loss_list[2][0].detach()#{0:'L', 1:'R'}
                #contact state
                _, contact_indices = torch.max(contact_vector, 2)
                contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()# becaome 0-4
                #hand
                hand = torch.sigmoid(lr_vector) > 0.5
                hand = hand.squeeze(0).float()#{0:'L', 1:'R'}
                
                #bbx    
                boxes = rois.data[:, :, 1:5]; # could be one of ['__background__', 'targetobject', 'hand']
                scores = cls_prob.data;#['__background__', 'targetobject', 'hand']  
                
                pascal_classes= np.asarray(['__background__', 'targetobject', 'hand']);  
                if cfg.TEST.BBOX_REG:
                    # Apply bounding-box regression deltas
                    box_deltas = bbox_pred.data
                    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                      if args.class_agnostic:
                          if args.cuda > 0:
                              box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                          else:
                              box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
            
                          box_deltas = box_deltas.view(1, -1, 4)
                      else:
                          if args.cuda > 0:
                              box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                          else:
                              box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                          box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))
            
                    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
                else:
                    # Simply repeat the boxes, once for each class
                    pred_boxes = np.tile(boxes, (1, scores.shape[1]))    
                
                scores = scores.squeeze()
                pred_boxes /= im_scales[0]
                pred_boxes = pred_boxes.squeeze()
                
                hand_dets = None
                hand_index=[];
                for j in range(1, len(pascal_classes)):
                    if pascal_classes[j] == 'hand':
                      inds = torch.nonzero(scores[:,j]>0.5).view(-1)
                    elif pascal_classes[j] == 'targetobject':
                      inds = torch.nonzero(scores[:,j]>0.5).view(-1)
                
                    # if there is det
                    if inds.numel() > 0:
                      cls_scores = scores[:,j][inds]
                      _, order = torch.sort(cls_scores, 0, True)
                      if args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                      else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                        
                      cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], hand[inds]), 1)
                      cls_dets = cls_dets[order]
                      keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                      cls_dets = cls_dets[keep.view(-1).long()]
                      if pascal_classes[j] == 'hand':
                        hand_dets = cls_dets.cpu().numpy()    
                        hand_index.append(inds[keep])
                        
                if hand_dets is not None:
                    for hand_idx, i in enumerate(range(np.minimum(10, hand_dets.shape[0]))):
                        hand_id=int(hand_dets[i, -1]);
                        hand_bbox = list(int(np.round(x)) for x in hand_dets[i, :4])
                        hand_info_frame.append([hand_id,hand_bbox]);
                        
                for k in range(len(hand_info_frame)):
                    hand_idx=hand_info_frame[k][0];#hand id: {0:'L', 1:'R'}
                    idx=hand_index[0][k];# the id of ROI
                    fea=handrole_features[idx];
            
                    label_file=open(self.labeling_path, "r")
                    for line in label_file:
                        current_image=line.split(' - ')[0];
                        hand=line.split(' - ')[1][0];
                        current_label=line.split(': ')[1][0];#{0:'L', 1:'R'}
                        if int(current_label)==2:
                            current_label=0;
                        if hand_idx==0:
                            current_hand='L';
                        elif hand_idx==1:
                            current_hand='R';
                        if (img_name==current_image) and (hand==current_hand):
                            self.imagename.append(img_name)
                            self.hand_info.append(hand_info_frame[k])
                            self.handrole_fea.append(fea);
                            self.labels.append(torch.tensor([int(current_label)]));
                            
                    label_file.close()
                    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):      
        imagename=self.imagename[idx];
        label=self.labels[idx];
        handrole_fea=self.handrole_fea[idx];  
        hand_info=self.hand_info[idx];        
        
        summary=[imagename, hand_info, handrole_fea, label];
        
        return summary

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--cag', dest='class_agnostic',
                    help='whether perform class_agnostic bbox regression',
                    action='store_true')
  parser.add_argument('--cuda', dest='cuda', 
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/res101.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--mode', dest='set_mode',
                      help='Which subdataset to load and save. range(0,N)', default=None,type=int)
  parser.add_argument('--load_set', dest='dataset_mode',
                      help='train or val or tetst set to load', default=None,type=str)
  parser.add_argument('--num_split', dest='num_split',
                      help='split the dataset into N pieces', default=1,type=int)
  parser.add_argument('--subj', dest='subj',
                      help='Which subject', default='',type=int)
  
  args = parser.parse_args()
  return args
args = parse_args()

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def Get_img_path(image_folder, labeling_txt_path, mode):
    set_folder=''
    image_list=[];
    labeling_file=open(labeling_txt_path, "r")
    num_instances=len(labeling_file.readlines());
    labeling_file.close()
    if args.num_split==1:
        for i in range(1,num_instances):
            current_image=linecache.getline(labeling_txt_path,i).split(' - ')[0];
            dset=current_image[0:2];
            subj=current_image[2:4];
            vset=current_image[6:8];
            if dset=='Dd':#homelab
                set_folder='frames_HomeLab';
            elif dset=='Ed':#Home
                set_folder='frames_Home';
                
            image_path=image_folder+'/'+set_folder+'/'+subj+'/'+vset+'/'+current_image;
            if image_path not in image_list:
                image_list.append(image_path)
    
    else:
        mode_set=[1];
        num_split=args.num_split
        for i in range(1,num_split+1):
            mode_set.append(i/num_split)
        if mode==0:
            for i in range(mode_set[mode],round(num_instances*mode_set[mode+1])):
                current_image=linecache.getline(labeling_txt_path,i).split(' - ')[0];
                dset=current_image[0:2];
                subj=current_image[2:4];
                vset=current_image[6:8];
                if dset=='Dd':#homelab
                    set_folder='frames_HomeLab';
                elif dset=='Ed':#Home
                    set_folder='frames_Home';
                    
                image_path=image_folder+'/'+set_folder+'/'+subj+'/'+vset+'/'+current_image;
                if image_path not in image_list:
                    image_list.append(image_path) 
        for i in range(round(num_instances*mode_set[mode]),round(num_instances*mode_set[mode+1])):
            current_image=linecache.getline(labeling_txt_path,i).split(' - ')[0];
            dset=current_image[0:2];
            subj=current_image[2:4];
            vset=current_image[6:8];
            if dset=='Dd':#homelab
                set_folder='frames_HomeLab';
            elif dset=='Ed':#Home
                set_folder='frames_Home';
                
            image_path=image_folder+'/'+set_folder+'/'+subj+'/'+vset+'/'+current_image;
            if image_path not in image_list:
                image_list.append(image_path)
    
    return image_list
##################### load hand object detector model ##############################
load_name = os.path.join('{path}/hand_object_detector/models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth')
pascal_classes = np.asarray(['__background__', 'targetobject', 'hand']) 
args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]'] 
# initilize the network here.
fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print('Using CUDA: ',device)

fasterRCNN.create_architecture()
checkpoint = torch.load(load_name)
fasterRCNN.load_state_dict(checkpoint['model'])
if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']

# initilize the tensor holder here.
im_data = torch.FloatTensor(1)
im_info = torch.FloatTensor(1)
num_boxes = torch.LongTensor(1)
gt_boxes = torch.FloatTensor(1)
box_info = torch.FloatTensor(1) 
# ship to cuda
if args.cuda > 0:
  im_data = im_data.cuda()
  im_info = im_info.cuda()
  num_boxes = num_boxes.cuda()
  gt_boxes = gt_boxes.cuda()

with torch.no_grad():
  if args.cuda > 0:
    cfg.CUDA = True

  if args.cuda > 0:
    fasterRCNN.cuda()

  fasterRCNN.to(device)
  fasterRCNN.eval()


###### geting a list of all targeted images
subject=str(args.subj);
label_folder='{path to label folder}/hand_object_detector/LOSOCV_manip_Home_HomeLab_labels/';
dataloader_folder='{path to data loader folder}/DataLoader_HandRole/';
image_folder='{path to raw images}/hand_object_detector/Datasets/';
flip='';# {'': no data augmentation,{'flip'}: data augmentation}
save_loader_name=dataloader_folder+'/Sub'+subject+'/Dataloader/Home_HomeLab_'+args.dataset_mode+'_loader_'+str(args.set_mode)+'_Sub'+subject+'_'+flip+'.pt';
labeling_txt_path_train=label_folder+'/LOSOCV_Home_HomeLab_sub'+subject+'_Manipulation_train_manip.txt';
labeling_txt_path_val=label_folder+'/LOSOCV_Home_HomeLab_sub'+subject+'_Manipulation_val.txt';
labeling_txt_path_test=label_folder+'/LOSOCV_Home_HomeLab_sub'+subject+'_Manipulation_test.txt';

if not os.path.exists(os.path.join(dataloader_folder,'Sub'+subject)):
    os.makedirs(os.path.join(dataloader_folder,'Sub'+subject));

if not os.path.exists(os.path.join(dataloader_folder,'Sub'+subject,'Dataloader')):
    os.makedirs(os.path.join(dataloader_folder,'Sub'+subject,'Dataloader'));
    
if args.dataset_mode=='train':
    labeling_txt_path=labeling_txt_path_train;
elif args.dataset_mode=='val':
    labeling_txt_path=labeling_txt_path_val;
elif args.dataset_mode=='test':
    labeling_txt_path=labeling_txt_path_test;
    
num_split=args.num_split;
print('num_split: ',num_split)

image_list=Get_img_path(image_folder, labeling_txt_path, mode=args.set_mode)
print('=== Subj ',subject,' ',args.dataset_mode,' Set. Split No. ',args.set_mode,' ===')
start_loading=datetime.now();
dataset=HandRoleDataset(image_path=image_list, labeling_path=labeling_txt_path,flip_mode=flip);
end_loading=datetime.now();
diff_time=(end_loading-start_loading).total_seconds();
print('Loading '+args.dataset_mode+' set takes: %.2f minutes' % (diff_time/60))
print('Number of instances: ',len(dataset))

start_loading=datetime.now();
torch.save(dataset, gzip.GzipFile(save_loader_name+'.gz','wb'))
end_loading=datetime.now();
diff_time=(end_loading-start_loading).total_seconds();
print('Saving dataset takes: %.2f minutes' % (diff_time/60))
