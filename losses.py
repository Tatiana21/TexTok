from math import log2
from functools import wraps, partial

import torch
from torch import nn, einsum, Tensor
from torch.nn import functional as F
from torch.nn import Module, ModuleList
from torch.autograd import grad as torch_grad
from torch.cuda.amp import autocast

import torchvision
from torchvision.models import VGG16_Weights

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from beartype import beartype
from beartype.typing import Tuple, List

from taylor_series_linear_attention import TaylorSeriesLinearAttn

from kornia.filters import filter3d
# helper

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def hinge_gen_loss(fake):
    return -fake.mean()

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

@autocast(enabled = False)
@beartype
def grad_layer_wrt_loss(
    loss: Tensor,
    layer: nn.Parameter
):
    return torch_grad(
        outputs = loss,
        inputs = layer,
        grad_outputs = torch.ones_like(loss),
        retain_graph = True
    )[0].detach()

# helper classes

def Sequential(*modules):
    modules = [*filter(exists, modules)]

    if len(modules) == 0:
        return nn.Identity()

    return nn.Sequential(*modules)

class Residual(Module):
    @beartype
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# rmsnorm

class RMSNorm(Module):
    def __init__(
        self,
        dim,
        channel_first = False,
        images = False,
        bias = False
    ):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(x, dim = (1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias

class AdaptiveRMSNorm(Module):
    def __init__(
        self,
        dim,
        *,
        dim_cond,
        channel_first = False,
        images = False,
        bias = False
    ):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.dim_cond = dim_cond
        self.channel_first = channel_first
        self.scale = dim ** 0.5

        self.to_gamma = nn.Linear(dim_cond, dim)
        self.to_bias = nn.Linear(dim_cond, dim) if bias else None

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)

        if bias:
            nn.init.zeros_(self.to_bias.weight)
            nn.init.zeros_(self.to_bias.bias)

    @beartype
    def forward(self, x: Tensor, *, cond: Tensor):
        batch = x.shape[0]
        assert cond.shape == (batch, self.dim_cond)

        gamma = self.to_gamma(cond)

        bias = 0.
        if exists(self.to_bias):
            bias = self.to_bias(cond)

        if self.channel_first:
            gamma = append_dims(gamma, x.ndim - 2)

            if exists(self.to_bias):
                bias = append_dims(bias, x.ndim - 2)

        return F.normalize(x, dim = (1 if self.channel_first else -1)) * self.scale * gamma + bias

# attention

class Attention(Module):
    @beartype
    def __init__(
        self,
        *,
        dim,
        dim_cond: int | None = None,
        causal = False,
        dim_head = 32,
        heads = 8,
        flash = False,
        dropout = 0.,
        num_memory_kv = 4
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.need_cond = exists(dim_cond)

        if self.need_cond:
            self.norm = AdaptiveRMSNorm(dim, dim_cond = dim_cond)
        else:
            self.norm = RMSNorm(dim)

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = heads)
        )

        assert num_memory_kv > 0
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_memory_kv, dim_head))

        self.attend = Attend(
            causal = causal,
            dropout = dropout,
            flash = flash
        )

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias = False)
        )

    @beartype
    def forward(
        self,
        x,
        mask: Tensor | None = None,
        cond: Tensor | None = None
    ):
        maybe_cond_kwargs = dict(cond = cond) if self.need_cond else dict()

        x = self.norm(x, **maybe_cond_kwargs)

        q, k, v = self.to_qkv(x)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = q.shape[0]), self.mem_kv)
        k = torch.cat((mk, k), dim = -2)
        v = torch.cat((mv, v), dim = -2)

        out = self.attend(q, k, v, mask = mask)
        return self.to_out(out)

class LinearAttention(Module):
    """
    using the specific linear attention proposed in https://arxiv.org/abs/2106.09681
    """

    @beartype
    def __init__(
        self,
        *,
        dim,
        dim_cond: int | None = None,
        dim_head = 8,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.need_cond = exists(dim_cond)

        if self.need_cond:
            self.norm = AdaptiveRMSNorm(dim, dim_cond = dim_cond)
        else:
            self.norm = RMSNorm(dim)

        self.attn = TaylorSeriesLinearAttn(
            dim = dim,
            dim_head = dim_head,
            heads = heads
        )

    def forward(
        self,
        x,
        cond: Tensor | None = None
    ):
        maybe_cond_kwargs = dict(cond = cond) if self.need_cond else dict()

        x = self.norm(x, **maybe_cond_kwargs)

        return self.attn(x)

class LinearSpaceAttention(LinearAttention):
    def forward(self, x, *args, **kwargs):
        x = rearrange(x, 'b c ... h w -> b ... h w c')
        x, batch_ps = pack_one(x, '* h w c')
        x, seq_ps = pack_one(x, 'b * c')

        x = super().forward(x, *args, **kwargs)

        x = unpack_one(x, seq_ps, 'b * c')
        x = unpack_one(x, batch_ps, '* h w c')
        return rearrange(x, 'b ... h w c -> b c ... h w')

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = 1)
        return F.gelu(gate) * x

class FeedForward(Module):
    @beartype
    def __init__(
        self,
        dim,
        *,
        dim_cond: int | None = None,
        mult = 4,
        images = True
    ):
        super().__init__()
        conv_klass = nn.Conv2d if images else nn.Conv3d

        rmsnorm_klass = RMSNorm if not exists(dim_cond) else partial(AdaptiveRMSNorm, dim_cond = dim_cond)

        maybe_adaptive_norm_klass = partial(rmsnorm_klass, channel_first = True, images = images)

        dim_inner = int(dim * mult * 2 / 3)

        self.norm = maybe_adaptive_norm_klass(dim)

        self.net = Sequential(
            conv_klass(dim, dim_inner * 2, 1),
            GEGLU(),
            conv_klass(dim_inner, dim, 1)
        )

    @beartype
    def forward(
        self,
        x: Tensor,
        *,
        cond: Tensor | None = None
    ):
        maybe_cond_kwargs = dict(cond = cond) if exists(cond) else dict()

        x = self.norm(x, **maybe_cond_kwargs)
        return self.net(x)

class Blur(Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(
        self,
        x,
        space_only = False,
        time_only = False
    ):
        assert not (space_only and time_only)

        f = self.f

        if space_only:
            f = einsum('i, j -> i j', f, f)
            f = rearrange(f, '... -> 1 1 ...')
        elif time_only:
            f = rearrange(f, 'f -> 1 f 1 1')
        else:
            f = einsum('i, j, k -> i j k', f, f, f)
            f = rearrange(f, '... -> 1 ...')

        is_images = x.ndim == 4

        if is_images:
            x = rearrange(x, 'b c h w -> b c 1 h w')

        out = filter3d(x, f, normalized = True)

        if is_images:
            out = rearrange(out, 'b c 1 h w -> b c h w')

        return out

class DiscriminatorBlock(Module):
    def __init__(
        self,
        input_channels,
        filters,
        downsample = True,
        antialiased_downsample = True
    ):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding = 1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding = 1),
            leaky_relu()
        )

        self.maybe_blur = Blur() if antialiased_downsample else None

        self.downsample = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
            nn.Conv2d(filters * 4, filters, 1)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)

        x = self.net(x)

        if exists(self.downsample):
            if exists(self.maybe_blur):
                x = self.maybe_blur(x, space_only = True)

            x = self.downsample(x)

        x = (x + res) * (2 ** -0.5)
        return x

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

class TotalLoss(Module):
    def __init__(self, cfg, device):
        super().__init__()

        self.recon_loss_weight = cfg['recon_loss_weight']
        self.perceptual_loss_weight = cfg['perceptual_loss_weight']
        self.use_vgg = cfg['use_vgg']

        self.vgg = None
        if self.use_vgg:
            vgg = torchvision.models.vgg16(
                weights = VGG16_Weights.DEFAULT
            )
            vgg.classifier = Sequential(*vgg.classifier[:-2])
            self.vgg = vgg
            self.vgg.to(device)
        # main flag for whether to use GAN at all
        self.use_gan = cfg['use_gan']

        discr_kwargs = dict(
            dim = 1,
            image_size = cfg['image_size'],
            channels = 3,
            max_dim = 512
        )

        self.discr = Discriminator(**discr_kwargs)
        self.discr.to(device)

        self.adversarial_loss_weight = cfg['adversarial_loss_weight']
        # self.grad_penalty_loss_weight = cfg.grad_penalty_loss_weight

        self.has_gan = self.use_gan and self.adversarial_loss_weight > 0.

    def forward(self, input_img, recon_img):
        recon_loss = F.mse_loss(input_img, recon_img)

        # perceptual loss

        if self.use_vgg:
            input_vgg_input = input_img
            recon_vgg_input = recon_img

            # if channels == 1:
            #     input_vgg_input = repeat(input_vgg_input, 'b 1 h w -> b c h w', c = 3)
            #     recon_vgg_input = repeat(recon_vgg_input, 'b 1 h w -> b c h w', c = 3)

            # elif channels == 4:
            #     input_vgg_input = input_vgg_input[:, :3]
            #     recon_vgg_input = recon_vgg_input[:, :3]

            input_vgg_feats = self.vgg(input_vgg_input)
            recon_vgg_feats = self.vgg(recon_vgg_input)

            perceptual_loss = F.mse_loss(input_vgg_feats, recon_vgg_feats)
        else:
            perceptual_loss = torch.tensor(0.)

        # get gradient with respect to perceptual loss for last decoder layer
        # needed for adaptive weighting

        # last_dec_layer = self.conv_out.conv.weight

        norm_grad_wrt_perceptual_loss = None

        # if self.training and self.use_vgg and self.has_gan:
        #     norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p = 2)

        # per-frame image discriminator

        if self.has_gan:

            fake_logits = self.discr(recon_img)
            gen_loss = hinge_gen_loss(fake_logits)

            adaptive_weight = 1.

            # if exists(norm_grad_wrt_perceptual_loss):
            #     norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p = 2)
            #     adaptive_weight = norm_grad_wrt_perceptual_loss / norm_grad_wrt_gen_loss.clamp(min = 1e-3)
            #     adaptive_weight.clamp_(max = 1e3)

            #     if torch.isnan(adaptive_weight).any():
            #         adaptive_weight = 1.
        else:
            gen_loss = torch.tensor(0.)
            adaptive_weight = 0.
    

        # calculate total loss

        total_loss = recon_loss \
            + perceptual_loss * self.perceptual_loss_weight \
            + gen_loss * adaptive_weight * self.adversarial_loss_weight

        return total_loss
