import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn.functional import threshold, normalize
from torchvision.utils import save_image
import src.utils as utils
from src.dataloader import DatasetSegmentation, collate_fn
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
import matplotlib.pyplot as plt

dataset_path = "./bottle_glass_dataset"

# Load SAM model
sam = build_sam_vit_b(checkpoint="sam_vit_b_01ec64.pth")
# Create SAM LoRA
sam_lora = LoRA_sam(sam,4)  
model = sam_lora.sam

# Process the dataset
processor = Samprocessor(model)
dataset = DatasetSegmentation(dataset_path, processor)

# Create a dataloader
train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Initialize optimize and Loss
optimizer = Adam(sam_lora.lora_vit.parameters(), lr=1e-5, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

num_epochs = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
model.train()


for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_dataloader):
      
      outputs = model(batched_input=batch,
            multimask_output=False)
      
      list_gt_msk, list_pred_msk, list_bbox = utils.get_list_masks(batch, outputs)
      utils.tensor_to_image(list_gt_msk, list_pred_msk, list_bbox)
      max_h, max_w = utils.get_max_size(batch)
      stk_gt_msk, stk_pred_msk = utils.pad_batch_mask(list_gt_msk, list_pred_msk, max_h, max_w)

      #utils.batch_to_tensor_mask(batch)
      loss = seg_loss(stk_gt_msk.float(), stk_pred_msk.float())

      # backward pass (compute gradients of parameters w.r.t. loss)
      loss.requires_grad = True
      optimizer.zero_grad()

      loss.backward()

      # optimize
      optimizer.step()
      epoch_losses.append(loss.item())
         

    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')

# Save the parameters of the model in safetensors format
sam_lora.save_lora_parameters("lora.safetensors")
