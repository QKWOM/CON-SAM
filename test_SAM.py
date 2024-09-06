import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import albumentations as A
from skimage.transform import resize
from PIL import Image
join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
#from segment_anything import samus_model_registry
import torch.nn.functional as F
import argparse
from torchvision.utils import save_image
from MyDataset_SAM import MyDataset
from segment_anything.utils.transforms import ResizeLongestSide
# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument("-task_name", type=str, default="-ViT-L")
parser.add_argument("-model_type", type=str, default="vit_l")
parser.add_argument(
            "-checkpoint", type=str, default="/opt/data/private/workspace/hzh/sam/base_model/sam_vit_l_0b3195.pth"
)
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="use wandb to monitor training"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")

#parser.add_argument("-img_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/AAAData_final_big/test/image/")
#parser.add_argument("-mask_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/AAAData_final_big/test/label/")

parser.add_argument("-img_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/left/test/image") ######big
parser.add_argument("-mask_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/left/test/label")######big

#parser.add_argument("-img_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/AAAA_big+small/test/image")
#parser.add_argument("-mask_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/AAAA_big+small/test/label")

#parser.add_argument("-img_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/all/yolov10/image/")
#parser.add_argument("-mask_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/all/yolov10/label/")

#parser.add_argument("-img_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/AAA4000/test/image/")
#parser.add_argument("-mask_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/AAA4000/test/label/")

#parser.add_argument("-img_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/AAA5000/test/image/")
#parser.add_argument("-mask_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/AAA5000/test/label/")

#parser.add_argument("-img_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/AAA6000/test/image/")
#parser.add_argument("-mask_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/AAA6000/test/label/")

#parser.add_argument("-img_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/AAAsplit/test/image")
#parser.add_argument("-mask_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/AAAsplit/test/label")

#parser.add_argument("-img_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/all/test_single/image/35-II/left/")
#parser.add_argument("-mask_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/all/test_single/label/35-II/left/")

parser.add_argument("-yolo_box_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/all/yolov10/labels/")
# train
parser.add_argument("-num_epochs", type=int, default=100)
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
    "--resume", type=str,
      default="",
    #default="",
    help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:1")
args = parser.parse_args()

device = torch.device(args.device)


# %% set up the model
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
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.mask_decoder.parameters():
            param.requires_grad = False
        
        if hasattr(self.image_encoder, 'prompt_generator'):
            for param in self.image_encoder.prompt_generator.parameters():
                param.requires_grad = False
        if hasattr(self.image_encoder, 'input_Adapter'):
            for param in self.image_encoder.input_Adapter.parameters():
                param.requires_grad = False
        if hasattr(self.image_encoder, 'Space_Adapter'):
            for param in self.image_encoder.Space_Adapter.parameters():
                param.requires_grad = False
        if hasattr(self.image_encoder, 'MLP_Adapter'):
            for param in self.image_encoder.MLP_Adapter.parameters():
                param.requires_grad = False
        

    def forward(self, image, box, pt):
        
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)
                
        res = ResizeLongestSide(1024)
        box_torch = res.apply_boxes_torch(boxes = box_torch, original_size = image.shape[2:4])
        
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
        )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
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
import time
# %% the check function
def check(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou = 0
    smooth = 1e-8
    accuracy_list = []
    precision_list = []
    recall_list = []
    dice_list = []
    flag = 0
    with torch.no_grad():
        #for step, (img, mask,boxes, image_path) in enumerate(tqdm(loader)):
        sum = 0
        for step, (datapack) in enumerate(tqdm(loader)):
            #boxes_np = boxes.detach().cpu().numpy()
            
            image = datapack['image'].to(dtype = torch.float32, device=device)
            mask = datapack['mask'].to(dtype = torch.float32, device=device)
            bbox = torch.as_tensor(datapack['bboxes'], dtype=torch.float32, device=device)
            pt = get_click_prompt(datapack)
            image_path = datapack['img_path']


            save_path = image_path[0].replace('image','new')
                          
            s = time.time()
            preds = model(image, bbox, pt)
            e = time.time()
            sum = sum + (e-s)
            
            preds = (preds > 0.5).float()
            num_correct += (preds == mask).sum()
            num_pixels += torch.numel(preds)
            if preds.shape[0] == 2:
              mask2 = mask.clone()
              mask = torch.cat((mask, mask2), dim=0)            
            #if preds.shape[0] == 2:
              #preds = preds[0:1, :, :, :]
            #save_image(preds, save_path)
            
            
            intersection = torch.logical_and(preds, mask).sum().item()
            union = torch.logical_or(preds, mask).sum().item()
            iou += intersection / union if union > 0 else 0.0
            # dice_score += (2 * (preds * mask).sum()) / (
                    # (preds + mask).sum() + 1e-8
            # )
            true_positive = torch.logical_and(mask,preds)
            false_negative = torch.logical_and(mask,torch.logical_not(preds))
            true_negative = torch.logical_and(torch.logical_not(preds),torch.logical_not(mask))
            false_positive = torch.logical_and(torch.logical_not(mask),preds)

            recall = true_positive.sum()/(true_positive.sum()+false_negative.sum())
            recall_list.append(recall)

            accuracy = (true_positive.sum()+true_negative.sum())/(true_positive.sum()+true_negative.sum()+false_positive.sum()+false_negative.sum())
            accuracy_list.append(accuracy)

            precision = true_positive.sum()/(true_positive.sum()+false_positive.sum()+smooth)
            precision_list.append(precision)

            dice = (2*true_positive.sum()+smooth)/(2*true_positive.sum()+false_negative.sum()+false_positive.sum()+smooth)
            dice_list.append(dice)

    print(len(loader)/(sum*1.0))    
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}"
    )
    print(f"IoU score :{iou / len(loader)}")
    # print(f"Dice score:{dice_score / len(loader)}")
    
    print('accuracy:',torch.mean(torch.Tensor(accuracy_list)))
    print('precision:',torch.mean(torch.Tensor(precision_list)))
    print('recall:',torch.mean(torch.Tensor(recall_list)))
    print('dice:',torch.mean(torch.Tensor(dice_list)))

# %% the main check_accuracy function
def main():
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    #medsam_model = sam_model.to(device)
    medsam_model.eval()
 
    # %% load the pretrained model checkpoint
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("yes")
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            medsam_model.load_state_dict(checkpoint["model"])


    valid_transform = A.Compose(
        [
            A.Resize(height=1024, width=1024, interpolation=cv2.INTER_AREA),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
        ],
    )
    valid_dataset = MyDataset(args.img_dir, args.mask_dir, args.yolo_box_dir, transform=valid_transform)

    print("Number of training samples: ", len(valid_dataset))
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    check(valid_dataloader, medsam_model)


if __name__ == "__main__":
    main()
