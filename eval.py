import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

from model import TexTok
from train import COCODataset
import numpy



if __name__== "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device(1)
    
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    model = TexTok(patch_size = 8, #8 for image size 256, and 16 for image size 512
                    batch_size = batch_size, 
                    image_size = image_size,
                    hidden_size = 768, 
                    latent_dim = 8,
                    num_tokens = 128,
                    ViT_number_of_heads = 12, 
                    ViT_number_of_layers = 12,
                    device = device)

    checkpoint_path = "/home/tchoudha/Textok/checkpoints/checkpoint_epoch_10.pth"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    images, captions = next(iter(dataloader))
    images =images.to(device)

    _, recon_images = model({"image":images, "text":captions})

    output_dir = "output_images/"  # Directory to save images
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
    for i in range(recon_images.size(0)):  # recon_images.size(0) is the batch size
        recon_img_array = recon_images[i].detach().cpu().numpy()  # Detach from graph and move to CPU
        recon_img_array = (recon_img_array * 255).astype('uint8').transpose(1, 2, 0)
        recon_img_pil = Image.fromarray(recon_img_array)
        recon_img_pil.save(f"{output_dir}recon_image_{i + 1}.png")

        input_img_array = images[i].detach().cpu().numpy()
        input_img_array = (input_img_array * 255).astype('uint8').transpose(1, 2, 0)
        input_img_pil = Image.fromarray(input_img_array)
        input_img_pil.save(f"{output_dir}input_image_{i + 1}.png")