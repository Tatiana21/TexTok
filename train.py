import os
import yaml
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as dset

from pycocotools.coco import COCO
from PIL import Image

from model import TexTok
from losses import TotalLoss

from tqdm import tqdm

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file, transform=None, tokenizer_model='t5-base', device='cpu'):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.transform = transform
        self.device = device
        self.tokenizer_model = tokenizer_model
        self.img_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        caption = anns[0]['caption'] if anns else ""
        
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = f"{self.img_dir}/{img_info['file_name']}"
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, caption

def get_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(torch.pi * (current_step - warmup_steps) / (total_steps - warmup_steps))))
    
    return LambdaLR(optimizer, lr_lambda)

def update_ema_model(ema_model, model, decay):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

def train(cfg, model, dataloader, optimizer, scheduler, criterion, device):
    num_epochs=cfg['num_epochs']
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.to(device)
    model.train()

    ema_model = copy.deepcopy(model)
    ema_model.to(device)

    for epoch in range(num_epochs):
        total_loss = 0.0
        print("starting epoch", epoch)
        for step, (images, captions) in enumerate(tqdm(dataloader)):
            # pdb.set_trace()
            images =images.to(device)
            optimizer.zero_grad()
            
            _, reconstructed_images = model({"image":images, "text":captions})
            
            loss = criterion(reconstructed_images, images)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % cfg['EMA_every_step'] == 0:
                update_ema_model(ema_model, model, cfg['EMA_model_decay_rate'])
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'ema_model' : ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    batch_size = cfg['batch_size']
    image_size = cfg['image_size']

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    dataset = COCODataset(img_dir="/home/tchoudha/coco2017/train2017",
                            ann_file="/home/tchoudha/coco2017/annotations/captions_train2017.json", 
                            transform=transform,
                            device = device) #COCODataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    model = TexTok( patch_size = cfg['patch_size'], #8 for image size 256, and 16 for image size 512
                    batch_size = batch_size, 
                    image_size = image_size,
                    hidden_size = cfg['ViT_hidden_size'], 
                    latent_dim = cfg['latent_dim'],
                    num_tokens = cfg['num_tokens'],
                    ViT_number_of_heads = cfg['ViT_number_of_heads'], 
                    ViT_number_of_layers = cfg['ViT_number_of_layers'],
                    device = device )

    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.99))
    scheduler = get_scheduler(optimizer, warmup_steps=cfg['warmup_steps'], total_steps=len(dataloader) * cfg['num_epochs'])
    criterion = TotalLoss(cfg, device)#nn.MSELoss()
    
    train(cfg, model, dataloader, optimizer, scheduler, criterion, device)
