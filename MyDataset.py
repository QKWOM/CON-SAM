# -*- coding: utf-8 -*-

import os 
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import re
import cv2

def random_click(mask, class_id=1):
    indices = np.argwhere(mask == class_id)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt = indices[np.random.randint(len(indices))]
    return pt[np.newaxis, :], [point_label]

class MyDataset(Dataset):
    def __init__(self,image_dir,mask_dir,yolo_box_dir,transform=None,bbox_shift=5):
        
        #for x in os.listdir(mask_dir):
          #path = os.path.join(mask_dir,x)
          #img = cv2.imread(path)
          #h, w = img.shape[:2]
          #flag = 1
          #for y in range(h):
              #if flag == 0:
                  #break
              #for x in range(w):
                  #b, g, r = img[y,x]
                  #if r==255 and g==157 and b==151:
                      #flag = 0
                      #break
          #if flag == 1:
          #   os.remove(path)
          #    path2 = path.replace('label','image')
          #    os.remove(path2)
            
            
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.box_dir = yolo_box_dir
        self.transform = transform
        #SAM
        #self.image = os.listdir(image_dir)
        #yolo
        self.image = os.listdir(image_dir)
        self.bbox_shift = bbox_shift
        

    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir,self.image[index])
        mask_path = os.path.join(self.mask_dir,self.image[index])

        img_path = re.sub(r'\(\d+\)', '', img_path)
        mask_path = re.sub(r'\(\d+\)', '', mask_path)
        
        #print(self.image[index])

        #yolo
        box_path = os.path.join(self.box_dir,self.image[index])
        yolo_box_path = box_path.replace(".png",".txt")
        #yolo_box_path = yolo_box_path.replace("left","all")
        #yolo_box_path = yolo_box_path("image","labels")
        #mask_path = mask_path.replace(".txt",".png")


        image = np.array(Image.open(img_path).convert("RGB")) #RGB
        mask = np.array(Image.open(mask_path).convert("L"),dtype=np.float32)

        mask2 = np.array(Image.open(mask_path).convert("RGB"))
        mask[mask == 255.0] = 1.0
        
        #yolo box
        
        t_x = 0;
        t_y = 0
        t_w = 0;
        t_h = 0
            
        h, w = mask.shape

        with open(yolo_box_path, 'r') as f:
            first_line = f.readline().strip()
            split_strings = first_line.split()
            t_x = float(split_strings[1])
            t_y = float(split_strings[2])
            t_w = float(split_strings[3])
            t_h = float(split_strings[4])

        x_min = int((t_x * 2 * w - t_w * w) / 2) * 2
        y_min = int((t_y * 2 * h - t_h * h) / 2) * 2
        x_max = int((t_w * w + t_x * 2 * w) / 2) * 2
        y_max = int((t_h * h + t_y * 2 * h) / 2) * 2
        
        #在transform中将totensor省略
        if self.transform is not None:
            augmentation = self.transform(image=image,mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]
        image = np.transpose(image, (2, 0, 1))
        

        
        # sam box
        
        #y_indices, x_indices = np.where(mask > 0)
        #x_min, x_max = np.min(x_indices), np.max(x_indices)
        #y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # add perturbation to bounding box coordinates
        H, W = mask.shape
        x_min = max(0, x_min - self.bbox_shift)
        x_max = min(W, x_max + self.bbox_shift)
        y_min = max(0, y_min - self.bbox_shift)
        y_max = min(H, y_max + self.bbox_shift)
        bboxes = np.array([x_min, y_min, x_max, y_max])
        
        pt, point_label = random_click(np.array(mask), class_id=1)
        point_labels = np.array(point_label)
        
        #return (
        #    torch.tensor(image).float(),
        #    torch.tensor(mask[None, :, :]).float(),
        #    torch.tensor(bboxes).float(),
        #    
        #    img_path
        #)
        return{
             'image': image,
             'mask': torch.tensor(mask[None, :, :]).float(),
             'bboxes': bboxes,
             'img_path':img_path,
             'pt': pt,
             'p_label': point_labels
             }
