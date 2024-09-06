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
from MyDataset import MyDataset

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument("-task_name", type=str, default="-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str, default="/opt/data/private/workspace/hzh/sam/all_data_exam/frozen_image_true+input_a+pro_ge/best.pth"
)
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="use wandb to monitor training"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
#parser.add_argument("-img_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/all/test/image")
#parser.add_argument("-mask_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/all/test/label")
parser.add_argument("-img_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/all/test/image/")
parser.add_argument("-mask_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/all/test/label/")
parser.add_argument("-yolo_box_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/all/yolov10_unfog/labels")
# train
parser.add_argument("-num_epochs", type=int, default=1000)
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
parser.add_argument("--device", type=str, default="cuda:0")
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

    def forward(self, image, box, pt):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

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
    with torch.no_grad():
        #for step, (img, mask,boxes, image_path) in enumerate(tqdm(loader)):
        for step, (datapack) in enumerate(tqdm(loader)):
            #boxes_np = boxes.detach().cpu().numpy()
            #image, mask = img.to(device), mask.to(device)
            
            image = datapack['image'].to(dtype = torch.float32, device=device)
            mask = datapack['mask'].to(dtype = torch.float32, device=device)
            bbox = torch.as_tensor(datapack['bboxes'], dtype=torch.float32, device=device)
            pt = get_click_prompt(datapack)
            image_path = datapack['img_path']
            #print(image_path[0])
            #yolo
            y_true_path = image_path[0].replace('image','label')
            #print(y_true_path)
            #y_true_path = y_true_path.replace('yolov10','test')
            # print(y_true_path)
            mask2 = np.array(Image.open(y_true_path).convert("L"),dtype=np.float32)
            mask2[mask2 == 255.0] = 1.0
            mask2 = cv2.resize(mask2,(1024,1024),interpolation=cv2.INTER_LINEAR)
            mask1 = torch.tensor(mask2[None, :, :]).long()
            mask1 = mask1.to(device)
            mask = mask1
            
            # SAM
            save_path = image_path[0].replace('image','conver_left')
            
            preds = model(image, bbox, pt)
            preds = (preds > 0.5).float()
            num_correct += (preds == mask).sum()
            num_pixels += torch.numel(preds)
            
            #save predict image
            # yolo
            #save_path = y_true_path.replace("label","yolo_all")
            #save_image(preds,save_path)
            
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

            precision = true_positive.sum()/(true_positive.sum()+false_positive.sum())
            precision_list.append(precision)

            dice = (2*true_positive.sum()+smooth)/(2*true_positive.sum()+false_negative.sum()+false_positive.sum()+smooth)
            dice_list.append(dice)

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
    medsam_model.eval()


    checkpoint = torch.load(args.checkpoint, map_location=device)
    medsam_model.load_state_dict(checkpoint["model"], strict=False)

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
