import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as dset
from pycocotools.coco import COCO
from PIL import Image
from model import TexTok
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

def train(model, dataloader, optimizer, criterion, device, num_epochs=10):
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        print("starting epoch", epoch)
        for images, captions in tqdm(dataloader):
            # pdb.set_trace()
            images =images.to(device)
            optimizer.zero_grad()
            
            _, reconstructed_images = model({"image":images, "text":captions})
            loss = criterion(reconstructed_images, images)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 16
    image_size = 256

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    dataset = COCODataset(img_dir="/home/tchoudha/coco2017/val2017",
                            ann_file="/home/tchoudha/coco2017/annotations/captions_val2017.json", 
                            transform=transform,
                            device = device) #COCODataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    model = TexTok( patch_size = 16, #8 for image size 256, and 16 for image size 512
                    batch_size = batch_size, 
                    image_size = image_size,
                    hidden_size = 768, 
                    latent_dim = 8,
                    num_tokens = 128,
                    ViT_number_of_heads = 12, 
                    ViT_number_of_layers = 12,
                    device = device )
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    train(model, dataloader, optimizer, criterion, device, num_epochs=10)
