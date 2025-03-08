import torch
from torch.nn import functional as F
from torch.nn import Module
from torchvision.models import VGG16_Weights

class Discriminator(Module):
    @beartype
    def __init__(
        self,
        *,
        dim,
        image_size,
        channels = 3,
        max_dim = 512,
        attn_heads = 8,
        attn_dim_head = 32,
        linear_attn_dim_head = 8,
        linear_attn_heads = 16,
        ff_mult = 4,
        antialiased_downsample = False
    ):
        super().__init__()
        image_size = pair(image_size)
        min_image_resolution = min(image_size)

        num_layers = int(log2(min_image_resolution) - 2)

        blocks = []

        layer_dims = [channels] + [(dim * 4) * (2 ** i) for i in range(num_layers + 1)]
        layer_dims = [min(layer_dim, max_dim) for layer_dim in layer_dims]
        layer_dims_in_out = tuple(zip(layer_dims[:-1], layer_dims[1:]))

        blocks = []
        attn_blocks = []

        image_resolution = min_image_resolution

        for ind, (in_chan, out_chan) in enumerate(layer_dims_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(layer_dims_in_out) - 1)

            block = DiscriminatorBlock(
                in_chan,
                out_chan,
                downsample = is_not_last,
                antialiased_downsample = antialiased_downsample
            )

            attn_block = Sequential(
                Residual(LinearSpaceAttention(
                    dim = out_chan,
                    heads = linear_attn_heads,
                    dim_head = linear_attn_dim_head
                )),
                Residual(FeedForward(
                    dim = out_chan,
                    mult = ff_mult,
                    images = True
                ))
            )

            blocks.append(ModuleList([
                block,
                attn_block
            ]))

            image_resolution //= 2

        self.blocks = ModuleList(blocks)

        dim_last = layer_dims[-1]

        downsample_factor = 2 ** num_layers
        last_fmap_size = tuple(map(lambda n: n // downsample_factor, image_size))

        latent_dim = last_fmap_size[0] * last_fmap_size[1] * dim_last

        self.to_logits = Sequential(
            nn.Conv2d(dim_last, dim_last, 3, padding = 1),
            leaky_relu(),
            Rearrange('b ... -> b (...)'),
            nn.Linear(latent_dim, 1),
            Rearrange('b 1 -> b')
        )
    def forward(self, x):

        for block, attn_block in self.blocks:
            x = block(x)
            x = attn_block(x)

        return self.to_logits(x)

# modulatable conv from Karras et al. Stylegan2
# for conditioning on latents

def total_loss(Module):
    # GANloss

    # l2 reconstruction loss
    recon_loss = F.mse_loss(input_img, recon_img)

    # perceptual loss
    vgg_weights = VGG16_Weights.DEFAULT
    vgg = torchvision.models.vgg16(
                    weights = vgg_weights
                )

    vgg.classifier = Sequential(*vgg.classifier[:-2])

    self.vgg = vgg
    input_vgg_feats = self.vgg(input_vgg_input)
    recon_vgg_feats = self.vgg(recon_vgg_input)

    perceptual_loss = F.mse_loss(input_vgg_feats, recon_vgg_feats)

    # LeCAM loss

    total_loss = recon_loss * recon_loss_weight #\
                # + aux_losses * self.quantizer_aux_loss_weight \
                # + perceptual_loss * self.perceptual_loss_weight \
                # + gen_loss * adaptive_weight * adversarial_loss_weight
