import os

import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam, AdamW
from torch.nn.functional import threshold, normalize
from torchvision.utils import save_image
import src.utils as utils
from src.dataloader import DatasetSegmentation, collate_fn, SemSegDataset
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
import matplotlib.pyplot as plt
import yaml
import torch.nn.functional as F

from src.segment_anything.build_sam_sp import build_sam_sp_vit_b
from src.segment_anything.utils.Dice_KL import DiceCELossWithKL

"""
This file is used to train a LoRA_sam model. I use that monai DiceLoss for the training. The batch size and number of epochs are taken from the configuration file.
The model is saved at the end as a safetensor.

"""
# Load the config file
with open("./config.yaml", "r") as ymlfile:
    config_file = yaml.load(ymlfile, Loader=yaml.Loader)
alpha = 0.5
# Take dataset path
train_dataset_path = config_file["DATASET"]["TRAIN_PATH"]
# Load SAM model
sam_sp = build_sam_sp_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
#Create SAM LoRA
sam_lora = LoRA_sam(sam_sp, config_file["LORA"]["RANK"])
model = sam_lora.sam
# Process the dataset
processor = Samprocessor(model)
train_ds = DatasetSegmentation(config_file, processor, mode="train")
# train_ds = SemSegDataset(config_file["DATASET"]["TRAIN_PATH"], processor=processor,
#                          is_val=False, frac=config_file["TRAIN"]["FRAC"])
# print(len(train_ds.img_files), len(train_ds.mask_files))
# Create a dataloader
train_dataloader = DataLoader(train_ds, batch_size=config_file["TRAIN"]["BATCH_SIZE"], shuffle=True,
                              collate_fn=collate_fn)
# Initialize optimize and Loss
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0)
dis_seg_loss = DiceCELossWithKL(sigmoid=True, squared_pred=True, reduction='mean')  # 组合损失
num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# Set model to train and into the device
model.train()
model.to(device)


total_loss = []

for epoch in range(num_epochs):
    epoch_losses = []

    for i, batch in enumerate(tqdm(train_dataloader)):
        outputs = model(batched_input=batch,
                        multimask_output=False)

        stk_gt, stk_out, _stk_out = utils.stacking_batch_sp(batch, outputs)
        loss = dis_seg_loss(stk_out, _stk_out, stk_gt.float().to(device))
        optimizer.zero_grad()
        loss.backward()
        # optimize
        optimizer.step()
        epoch_losses.append(loss.item())

    print(f'EPOCH: {epoch}')
    print(f'Mean loss training: {mean(epoch_losses)}')

# Save the parameters of the model in safetensors format
rank = config_file["SAM"]["RANK"]
save_dir = os.path.join(config_file["SAVE"]["ROOT"], config_file["SAVE"]["NAME"])

os.makedirs(save_dir, exist_ok=True)
sam_lora.save_sam(os.path.join(str(save_dir), f"sam_{config_file['SAM']['TYPE']}.pt"))
sam_lora.save_lora_parameters(os.path.join(str(save_dir), f"lora_rank{rank}.safetensors"))
