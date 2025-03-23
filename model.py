import torch
from torch import nn
# import timm
# from timm.models.vision_transformer import VisionTransformer
from functools import partial
import math

from torchvision.models.vision_transformer import VisionTransformer, Encoder
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, pack, unpack
from transformers import T5Tokenizer, T5EncoderModel

import pdb

def divisible_by(num, den):
    return (num % den) == 0

def pack_square_height_width(t):
    assert t.ndim == 4
    return rearrange(t, 'b h w d -> b (h w) d')

def unpack_square_height_width(t):
    assert t.ndim == 3
    hw = int(math.sqrt(t.shape[1]))
    return rearrange(t, 'b (h w) d -> b h w d', h = hw, w = hw)


class TexTok(nn.Module):
    def __init__(self, patch_size: int, 
                        batch_size: int, 
                        image_size: int,
                        hidden_size: int,
                        latent_dim: int, 
                        num_tokens: int,
                        ViT_number_of_heads: int, 
                        ViT_number_of_layers: int,
                        device):
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size

        self.num_heads = ViT_number_of_heads
        self.depth = ViT_number_of_layers
        
        assert divisible_by(image_size, patch_size)

        self.num_patches = (image_size // patch_size) ** 2
        self.num_tokens = num_tokens
        self.num_text_tokens = 32
        self.text_token_dim = 512
        self.mlp_dim = 3072
        self.seq_length = self.num_patches + self.num_tokens + self.num_text_tokens #1184

        self.device = device

        self.image_to_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear(3 * patch_size * patch_size, hidden_size)
        )
        # self.patch_embed = nn.Conv2d(in_chans, hidden_size, kernel_size=patch_size, stride=patch_size)

        # Learnable image tokens (N x D)
        self.image_tokens = nn.Parameter(torch.randn(self.num_tokens, hidden_size))

        # Linear projection for text tokens
        self.text_proj_enc = nn.Linear(self.text_token_dim, hidden_size) 

        # Positional embeddings
        self.pos_emb = nn.Parameter(torch.zeros(self.seq_length, hidden_size))

        # Tokenizer (Encoder) ViT
        self.encoder = Encoder(
            seq_length = self.seq_length,
            num_layers = self.depth,
            num_heads = self.num_heads,
            hidden_dim = hidden_size,
            mlp_dim = self.mlp_dim,
            dropout = 0.0,
            attention_dropout = 0.0,
            norm_layer = partial(nn.LayerNorm, eps=1e-6),
        )

        # Linear projection to output image tokens (N x d)
        self.token_out_proj = nn.Linear(hidden_size, latent_dim)

        # Detokenizer (Decoder) ViT
        self.decoder = Encoder(
            seq_length = self.seq_length,
            num_layers = self.depth,
            num_heads = self.num_heads,
            hidden_dim = hidden_size,
            mlp_dim = self.mlp_dim,
            dropout = 0.0,
            attention_dropout = 0.0,
            norm_layer = partial(nn.LayerNorm, eps=1e-6),
        )
        # Learnable patch tokens (hw x D)
        self.patch_tokens = nn.Parameter(torch.randn(self.num_patches, hidden_size))
        
        # Linear projections for detokenizer
        self.image_token_proj = nn.Linear(latent_dim, hidden_size)
        self.text_proj_dec = nn.Linear(self.text_token_dim, hidden_size)
        
        # Reconstruction head

        self.tokens_to_image = nn.Sequential(
            nn.Linear(hidden_size, 3 * patch_size * patch_size),
            Rearrange('b h w (c p1 p2) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size)
        )

        nn.init.normal_(self.image_tokens, std = 0.02)
        nn.init.normal_(self.patch_tokens, std = 0.02)
        nn.init.normal_(self.pos_emb, std = 0.02)

        
    def text_embeder(self, text_caption, max_length=32, device = "cpu"):
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5EncoderModel.from_pretrained("t5-small").to(device)

        # enc = tokenizer(text_caption, return_tensors="pt")
        enc = tokenizer(text_caption, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)

        # forward pass through encoder only
        with torch.no_grad():
            encoded = model(**enc).last_hidden_state  # (B, Nt, D)

        return encoded

    def encode(self, input):
        # Tokenizer

        # 1) image patch tokens P ∈ R^hw×D from patchifying and flattening the input image with a projection layer
        img_patches = self.image_to_tokens(input["image"]) # B x h x w x D = 16 x 32 x 32 x 768
        img_patches = pack_square_height_width(img_patches) # B x hw x D = 16 x 1024 x 768

        # 2)  N randomly-initialized learnable image token 
        img_learnable = self.image_tokens.expand(input["image"].size(0), -1, -1) #B x N x D = 16 x N  x 768
        
        # 3) linearly projected text tokens
        text_embd = self.text_embeder(input["text"], max_length = self.num_text_tokens, device = self.device).to(self.device)
        text_proj_enc = self.text_proj_enc(text_embd)  # (B, Nt, D)

        tokenizer_input = torch.cat([img_patches, img_learnable, text_proj_enc], dim=1) #B x (hw + N + N_t) x D

        pos_emb = repeat(self.pos_emb, 'N D -> B N D', B = tokenizer_input.shape[0])
        
        tokenizer_input = tokenizer_input + pos_emb
        tokenizer_output = self.encoder(tokenizer_input)
        
        # Retain only the learned image tokens
        image_tokens = self.token_out_proj(tokenizer_output[:, self.num_patches:self.num_patches + self.num_tokens, :]) #B x N x d
        
        return image_tokens
    
    def decode(self, text, image_tokens):

        text_embd = self.text_embeder(text, max_length = self.num_text_tokens, device = self.device).to(self.device)
        
        patch_tokens = self.patch_tokens.expand(len(text), -1, -1) #B x hw x D
        image_token_proj = self.image_token_proj(image_tokens) #B x N x D
        text_proj_dec = self.text_proj_dec(text_embd) # (B, Nt, D)

        detokenizer_input = torch.cat([patch_tokens, image_token_proj, text_proj_dec], dim=1)  #B x (hw + N + N_t) x D
        detokenizer_output = self.decoder(detokenizer_input)

        # Retain only the learned patch tokens and reconstruct the image
        reconstructed_patches = detokenizer_output[:, :self.num_patches, :] #B x hw x D

        # reconstructed_img = self.reconstruction_head(reconstructed_patches.transpose(1, 2).reshape(img.shape[0], -1, self.h, self.w))
        reconstructed_patches = unpack_square_height_width(reconstructed_patches)
        reconstructed_img = self.tokens_to_image(reconstructed_patches) #B x H x W

        return reconstructed_img
    
    def forward(self, input):
        
        # Tokenizer
        image_tokens = self.encode(input) #B x N x d 

        # Detokenizer
        reconstructed_img = self.decode(input["text"], image_tokens) #B x H x W

        return image_tokens, reconstructed_img