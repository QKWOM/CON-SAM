# -*- coding: utf-8 -*-

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import re
import cv2


def get_white_regions_coordinates(gray):

    gray = gray.astype(np.uint8)
    gray[gray == 1] = 255
    W, H = gray.shape[0], gray.shape[1]
    
    #y_indices, x_indices = np.where(gray > 0)
    #x_min, x_max = np.min(x_indices), np.max(x_indices)
    #y_min, y_max = np.min(y_indices), np.max(y_indices)
    #return np.array([x_min, y_min, x_max, y_max])    
    
    # 应用阈值，找到白色区域
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # 查找白色区域的轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 用于存储左右两侧的白色区域
    left_region = None
    right_region = None
    bbox_shift = 4
    coordinates = []
    flag=0
    for contour in contours:
        # 计算每个轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        x_min, y_min = top_left[0], top_left[1]
        x_max, y_max = bottom_right[0], bottom_right[1]
        x_min = max(0, x_min - bbox_shift)
        x_max = min(W, x_max + bbox_shift)
        y_min = max(0, y_min - bbox_shift)
        y_max = min(H, y_max + bbox_shift)
        coordinates.append([x_min, y_min, x_max, y_max])
    coordinates_array = np.array(coordinates)
    if coordinates_array.shape[0]==1:
        return coordinates_array.flatten()
    else:
        return coordinates_array


def random_click(mask, class_id=1):
    indices = np.argwhere(mask == class_id)
    indices[:, [0, 1]] = indices[:, [1, 0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0, 1]] = indices[:, [1, 0]]
    pt = indices[np.random.randint(len(indices))]
    return pt[np.newaxis, :], [point_label]


class MyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, yolo_box_dir, transform=None, bbox_shift=2):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.box_dir = yolo_box_dir
        self.transform = transform
        # SAM
        self.image = os.listdir(image_dir)

        self.bbox_shift = bbox_shift

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image[index])
        mask_path = os.path.join(self.mask_dir, self.image[index])

        img_path = re.sub(r'\(\d+\)', '', img_path)
        mask_path = re.sub(r'\(\d+\)', '', mask_path)
        
        image = np.array(Image.open(img_path).convert("RGB"))  # RGB
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        
        #(b, g, r) = cv2.split(image)
        #bH = cv2.equalizeHist(b)
        #gH = cv2.equalizeHist(g)
        #rH = cv2.equalizeHist(r)
        #image = cv2.merge((bH, gH, rH))
        
        mask[mask == 255.0] = 1.0

        # 在transform中将totensor省略
        if self.transform is not None:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]
        image = np.transpose(image, (2, 0, 1))



        # two box or one box (bid picture)
        bboxes = get_white_regions_coordinates(mask)
        #if bboxes.shape[0] == 1:
            #bboxes = np.vstack([bboxes,bboxes])

        ### point prompt
        pt, point_label = random_click(np.array(mask), class_id=1)
        point_labels = np.array(point_label)
        ### point prompt
        
        
        return {
            'image': image,
            'mask': torch.tensor(mask[None, :, :]).float(),
            'bboxes': bboxes,
            'img_path': img_path,
            'pt': pt,
            'p_label': point_labels
        }


























