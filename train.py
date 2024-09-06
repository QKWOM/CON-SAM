# -*- coding: utf-8 -*-
# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import albumentations as A
import math
from typing import Tuple

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
from segment_anything.modeling.image_encoder import Block
from segment_anything.utils.transforms import ResizeLongestSide

# set seeds and 释放显存
torch.manual_seed(2023)
torch.cuda.empty_cache()

#img_dir = "/opt/data/private/workspace/hzh/all_data/AAAData_final_big/train/image"
#mask_dir = "/opt/data/private/workspace/hzh/all_data/AAAData_final_big/train/label"

#img_dir = "/opt/data/private/workspace/hzh/all_data/all/train/image"
#mask_dir = "/opt/data/private/workspace/hzh/all_data/all/train/label"

#img_dir = "/opt/data/private/workspace/hzh/all_data/AAAsplit/train/image"
#mask_dir = "/opt/data/private/workspace/hzh/all_data/AAAsplit/train/label"

# img_dir = "/opt/data/private/workspace/hzh/all_data/AAAA_big+small/train/image"
# mask_dir = "/opt/data/private/workspace/hzh/all_data/AAAA_big+small/train/label"

# img_dir = "/opt/data/private/workspace/hzh/all_data/AAA_not_clear/train/image"
# mask_dir = "/opt/data/private/workspace/hzh/all_data/AAA_not_clear/train/label"

# img_dir = "/opt/data/private/workspace/hzh/all_data/AAA4000/train/image/"
# mask_dir = "/opt/data/private/workspace/hzh/all_data/AAA4000/train/label/"

# img_dir = "/opt/data/private/workspace/hzh/all_data/AAA5000/train/image/"
# mask_dir = "/opt/data/private/workspace/hzh/all_data/AAA5000/train/label/"

# img_dir = "/opt/data/private/workspace/hzh/all_data/AAA6000/train/image/"
# mask_dir = "/opt/data/private/workspace/hzh/all_data/AAA6000/train/label/"

img_dir = "/opt/data/private/workspace/hzh/sam/all_data_exam/AAAsplit+ori8000/train/image/"
mask_dir = "/opt/data/private/workspace/hzh/sam/all_data_exam/AAAsplit+ori8000/train/label/"

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument("-task_name", type=str, default="SAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str,
    default="/opt/data/private/workspace/hzh/sam/all_data_exam/frozen_image_true+input_a+pro_ge/best.pth"
)
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="use wandb to monitor training"
)

parser.add_argument("-eval_img_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/AAAData_final_big/test/image")
parser.add_argument("-eval_mask_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/AAAData_final_big/test/label")

#parser.add_argument("-eval_img_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/all/test/image")
#parser.add_argument("-eval_mask_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/all/test/label")

# parser.add_argument("-eval_img_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/AAAA_big+small/test/image")
# parser.add_argument("-eval_mask_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/AAAA_big+small/test/label")

parser.add_argument("-yolo_box_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/all/yolov10/labels/")
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=200)
parser.add_argument("-batch_size", type=int, default=1)
parser.add_argument("-num_workers", type=int, default=1)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument("-use_amp", action="store_true", default=True, help="use amp")
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:1")
args = parser.parse_args()

# %% set up model for training
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
# model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
model_save_path = join("all_data_exam", args.task_name + "-" + run_id)
device = torch.device(args.device)


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


# %% set up model

class MedSAM(nn.Module):
    def __init__(
            self,
            image_encoder,
            mask_decoder,
            prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        # freeze image encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = True
        # freeze mask decoder
        for param in self.mask_decoder.parameters():
            param.requires_grad = True

        if hasattr(self.image_encoder, 'prompt_generator'):
            for param in self.image_encoder.prompt_generator.parameters():
                param.requires_grad = True
        if hasattr(self.image_encoder, 'input_Adapter'):
            for param in self.image_encoder.input_Adapter.parameters():
                param.requires_grad = True
        if hasattr(self.image_encoder, 'Space_Adapter'):
            for param in self.image_encoder.Space_Adapter.parameters():
                param.requires_grad = False
        if hasattr(self.image_encoder, 'MLP_Adapter'):
            for param in self.image_encoder.MLP_Adapter.parameters():
                param.requires_grad = False

    # 前向传播，进行推理
    def forward(self, image, box, pt):
        # 提取图像的特征
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():  # 不计算梯度（提示编码器）
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

        res = ResizeLongestSide(1024)
        box_torch = res.apply_boxes_torch(boxes=box_torch, original_size=image.shape[2:4])

        # 对框进行编码
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        # 对图像、框的嵌入进行解码
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        # 使用双线性插值对分辨率低的掩码进行上采样，得到与原始图像分辨率相同的掩码
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


def get_click_prompt(datapack):
    if 'pt' not in datapack:
        imgs, pt, masks = generate_click_prompt(imgs, masks)
    else:
        pt = datapack['pt']
        point_labels = datapack['p_label']

    point_coords = pt
    coords_torch = torch.as_tensor(point_coords, dtype=torch.float32, device=device)
    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)
    if len(pt.shape) == 2:
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
    pt = (coords_torch, labels_torch)
    return pt


def main():
    os.makedirs(model_save_path, exist_ok=True)  # exist_ok=True 在目录存在时不会报错
    # __file__是当前执行脚本的绝对路径
    # 将当前执行脚本文件复制到model_save_path目录下，并且给它一个run_id的前缀
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )
    # 加载模型，并使用指定的checkpoint
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)

    # 设置为训练模式
    medsam_model.train()

    params = list(medsam_model.image_encoder.parameters()) + list(medsam_model.prompt_encoder.parameters()) + list(
        medsam_model.mask_decoder.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay
    )

    start_epoch = 0

    checkpoint = torch.load(args.checkpoint, map_location=device)

    # start_epoch = checkpoint["epoch"] + 1

    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in params if p.requires_grad),
    )  # 93729252
    # 计算dice
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    # 二元交叉熵损失函数
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # 计算边界框之间的相似性的损失函数
    loss_fn = BoundaryDoULoss()
    # %% train
    # 数据预处理流程，包括调整图像大小和使用均值和标准差进行标准化
    train_transform = A.Compose(
        [
            A.Resize(height=1024, width=1024, interpolation=cv2.INTER_AREA),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
        ],
    )
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    best_loss = 1e10
    best_dice = 0
    train_dataset = MyDataset(img_dir, mask_dir, args.yolo_box_dir, transform=train_transform)

    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # %% load the pretrained model checkpoint
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"], strict=False)
            # print(optimizer.state_dict()["state"].keys())
            # print(medsam_model.state_dict().keys())
            # optimizer.load_state_dict(checkpoint["optimizer"])

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    # 根据epoch数进行训练
    N = 8

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0

        # for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
        for step, (datapack) in enumerate(tqdm(train_dataloader)):
            # 梯度清零，确保梯度不会背错误的累积到前一个步骤中
            optimizer.zero_grad()

            image = datapack['image'].to(dtype=torch.float32, device=device)
            gt2D = datapack['mask'].to(dtype=torch.float32, device=device)
            bbox = torch.as_tensor(datapack['bboxes'], dtype=torch.float32, device=device)
            pt = get_click_prompt(datapack)

            # 是否使用混合精度训练（AMP）
            if args.use_amp:
                # AMP 使用低精度（float16）加速训练并减少显存使用
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = medsam_model(image, bbox, pt)
                    
                    if medsam_pred.shape[0] == 2:
                      gt2D = gt2D.repeat(2, 1, 1, 1)
                    if medsam_pred.shape[0] == 3:
                      gt2D = gt2D.repeat(3, 1, 1, 1)
                    loss = loss_fn(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float()) + seg_loss(medsam_pred,gt2D) + FocalLoss(inputs=medsam_pred, targets=gt2D)
                    #print(loss)  
                    # print(seg_loss(medsam_pred,gt2D))
                    # focal+dice
                    # loss = seg_loss(medsam_pred,gt2D) + FocalLoss(inputs=medsam_pred,targets=gt2D)
                    # focal+dice+DoU
                    # loss = seg_loss(medsam_pred,gt2D) + FocalLoss(inputs=medsam_pred,targets=gt2D) + loss_fn(medsam_pred,gt2D)
                    # focal+dice+BCE
                    # loss = seg_loss(medsam_pred,gt2D) + FocalLoss(inputs=medsam_pred,targets=gt2D) + ce_loss(medsam_pred,gt2D)
                    # print(loss_fn(medsam_pred,gt2D))
                    # print(ce_loss(medsam_pred,gt2D.float()))
                    # print(seg_loss(medsam_pred,gt2D))
                    # print(FocalLoss(inputs=medsam_pred,targets=gt2D))
                    # loss = seg_loss(medsam_pred,gt2D) + loss_focal(medsam_pred,gt2D) + ce_loss(medsam_pred,gt2D.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            iter_num += 1

        epoch_loss /= step
        losses.append(epoch_loss)
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )

        os.makedirs(model_save_path, exist_ok=True)

        ## save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }

            torch.save(checkpoint, join(model_save_path, "ce+DoU.pth"))


        # %% plot loss
        plt.plot(losses)
        plt.title("DoU Loss + CE Loss + Dice Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, args.task_name + "train_loss.png"))
        plt.close()


if __name__ == "__main__":
    main()
