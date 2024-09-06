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
from MyDataset_new import MyDataset

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
#parser.add_argument("-img_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/left/test/image")
#parser.add_argument("-mask_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/left/test/label")
parser.add_argument("-img_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/all/ksy_right_png/")
parser.add_argument("-mask_dir", type=str, default="/opt/data/private/workspace/hzh/all_data/all/test/label/")
parser.add_argument("-yolo_box_dir", type=str, default="/opt/data/private/workspace/hzh/ultralytics/ultralytics/runs/detect/predict8/labels")
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
parser.add_argument("--device", type=str, default="cuda:2")
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

    def forward(self, image, box):
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

            image = datapack['image'].to(dtype = torch.float32, device=device)
            bbox = torch.as_tensor(datapack['bboxes'], dtype=torch.float32, device=device)
            image_path = datapack['img_path']
            
            # SAM
            save_path = image_path[0].replace('ksy_right_png','boundry')
            
            preds = model(image, bbox)
            preds = (preds > 0.5).float()
            
            save_image(preds,save_path)
            




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
