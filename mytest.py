import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import albumentations as A
join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import monai
from utils import BoundaryDoULoss
from myutils import FocalLoss
from MyDataset_SAM import MyDataset
import torchvision.models as models
from torchsummary import summary

#samus_dict = samus.state_dict()
#dict_trained = {k: v for k, v in sam_dict.items() if k in samus_dict}
device = "cuda:1"
checkpoint = torch.load("/opt/data/private/workspace/hzh/sam/all_data_exam/SAM-ViT-B-20240814-1136/best.pth", map_location=device)

with open("/opt/data/private/workspace/hzh/sam/require_infoma2.txt", 'w') as f:
  for k, v in checkpoint['model'].items():
    #if "input_Adapter" in k:
      #f.write(v)
      v = v.cpu()
      f.write(k+'\n') 
      #np.savetxt("/opt/data/private/workspace/hzh/sam/require_infoma.txt",k)
