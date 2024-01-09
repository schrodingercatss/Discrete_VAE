import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from einops import rearrange
import math

class ResBlock(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_dim, in_dim, kernel_size=1, padding=0)
        )
    
    def forward(self, x):
        return self.layers(x) + x


class Discrete_VAE(nn.Module):
    def __init__(self, 
                image_size=256,
                num_tokens=512,
                codebook_dim=512,
                num_layers=3,
                num_resnet_blocks=0,
                hidden_dim=64,
                tau=0.9):
        super().__init__()
        self.image_size = image_size
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        self.num_layers = num_layers
        self.num_resnet_blocks = num_resnet_blocks
        self.hidden_dim = hidden_dim
        self.tau = tau

        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        encoder_channels = [hidden_dim] * num_layers
        decoder_channels = list(reversed(encoder_channels))

        decoder_init_channel = codebook_dim if num_resnet_blocks == 0 else decoder_channels[0]
        decoder_channels = [decoder_init_channel, *decoder_channels]

        encoder_channels_io, decoder_channels_io = map(lambda t: list(zip(t[:-1], t[1:])), [encoder_channels, decoder_channels])

        encoder_layers = []
        decoder_layers = []

        for (encoder_in, encoder_out), (decoder_in, decoder_out) in zip(encoder_channels_io, decoder_channels_io):
            encoder_layers.extend([nn.Conv2d(encoder_in, encoder_out, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2)])
            decoder_layers.extend([nn.ConvTranspose2d(decoder_in, decoder_out, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2)])

        for _ in range(num_resnet_blocks):
            decoder_layers.insert(0, ResBlock(decoder_channels[1]))
            encoder_layers.append(ResBlock(encoder_channels[-1]))
        
        if num_resnet_blocks > 0:
            decoder_layers.insert(0, nn.Conv2d(codebook_dim, decoder_channels[1], kernel_size=1, padding=0))
        
        encoder_layers.append(nn.Conv2d(encoder_channels[-1], num_tokens, kernel_size=1, padding=0))
        decoder_layers.append(nn.Conv2d(decoder_channels[-1], 3, kernel_size=1, padding=0))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def decode(self, img_seq):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = w = int(math.sqrt(n))
        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h = h, w = w)
        images = self.decoder(image_embeds)
        return images


    def forward(self, x):
        logits = self.encoder(x).view(-1, self.hidden_dim, self.num_tokens)
        log_qy = F.log_softmax(logits, dim=-1)
        log_uniform = torch.full_like(log_qy, -torch.log(self.num_tokens))
        kld = F.kl_div(log_qy, log_uniform, reduction='batchmean', log_target=True)

        if self.training:
            soft_onehot = F.gumbel_softmax(logits, tau=self.tau, dim=-1, hard=False)
        else:
            soft_onehot = F.softmax(logits, dim=-1)
        
        sampled = torch.einsum("b n h w, n d -> b d h w", soft_onehot, self.codebook.weight)

        out = self.decoder(sampled)

        return out, kld