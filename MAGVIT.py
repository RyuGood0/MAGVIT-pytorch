import torch
from torch import nn

class ResBlockX(nn.Module):
    def __init__(self, X) -> None:
        super().__init__()

        self.groupNorm = nn.GroupNorm(X, X)
        self.swish = nn.SiLU()
        self.conv1 = nn.Conv3d(X, X, (3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(X, X, (3, 3, 3), padding=1)

    def forward(self, x):
        residual = x

        x = self.swish(self.groupNorm(x))
        x = self.conv1(x)
        x = self.swish(self.groupNorm(x))
        x = self.conv2(x)

        return x + residual

class ResBlockXY(nn.Module):
    def __init__(self, X, Y) -> None:
        super().__init__()

        self.groupNorm1 = nn.GroupNorm(X, X)
        self.swish = nn.SiLU()
        self.conv1 = nn.Conv3d(X, Y, (3, 3, 3), padding=1)

        self.groupNorm2 = nn.GroupNorm(Y, Y)
        self.conv2 = nn.Conv3d(Y, Y, (3, 3, 3), padding=1)

        self.resConv = nn.Conv3d(X, Y, (1, 1, 1))

    def forward(self, x):
        residual = self.resConv(x)

        x = self.swish(self.groupNorm1(x))
        x = self.conv1(x)
        x = self.swish(self.groupNorm2(x))
        x = self.conv2(x)

        return x + residual

class ResBlockDown(nn.Module):
    def __init__(self, X, Y) -> None:
        super().__init__()

        self.leakyRELU = nn.LeakyReLU()
        self.pool = nn.AvgPool3d((2, 2, 2), padding=1)
        self.conv1 = nn.Conv3d(X, Y, (3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(Y, Y, (3, 3, 3), padding=1)

        self.resConv = nn.Conv3d(X, Y, (1, 1, 1))

    def forward(self, x):
        residual = self.resConv(self.pool(x))

        x = self.leakyRELU(self.conv1(x))
        x = self.pool(x)
        x = self.leakyRELU(self.conv2(x))

        return x + residual

from vector_quantize_pytorch import VectorQuantize
from einops import rearrange
class MAGVQVAE(nn.Module):
    def __init__(self, c, codebook_size=1024, codebook_dim=256, threshold_ema_dead_code=2) -> None:
        super().__init__()

        self.vq = VectorQuantize(
            dim = codebook_dim,
            codebook_size = codebook_size,
            use_cosine_sim = True,
            threshold_ema_dead_code = threshold_ema_dead_code
        )

        self.encoder = nn.Sequential(
            nn.Conv3d(3, 64*c, (3, 3, 3), padding=1),
            ResBlockX(64*c),
            ResBlockX(64*c),
            nn.AvgPool3d((2, 2, 2)),
            ResBlockXY(64*c, 128*c),
            ResBlockX(128*c),
            nn.AvgPool3d((2, 2, 2)),
            ResBlockX(128*c),
            ResBlockX(128*c),
            nn.AvgPool3d((1, 2, 2)),
            ResBlockXY(128*c, 256*c),
            ResBlockX(256*c),
            ResBlockX(256*c),
            ResBlockX(256*c),
            nn.GroupNorm(256*c, 256*c),
            nn.SiLU(),
            nn.Conv3d(256*c, 256, (1, 1, 1))
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(256, 256*c, (3, 3, 3), padding=1),
            ResBlockX(256*c),
            ResBlockX(256*c),
            ResBlockX(256*c),
            ResBlockX(256*c),
            nn.Upsample(scale_factor=(1, 2, 2), mode='nearest'),
            nn.Conv3d(256*c, 256, (3, 3, 3), padding=1),
            ResBlockXY(256, 128*c),
            ResBlockX(128*c),
            nn.Upsample(scale_factor=(2, 2, 2), mode='nearest'),
            nn.Conv3d(128*c, 128*c, (3, 3, 3), padding=1),
            ResBlockX(128*c),
            ResBlockX(128*c),
            nn.Upsample(scale_factor=(2, 2, 2), mode='nearest'),
            nn.Conv3d(128*c, 128*c, (3, 3, 3), padding=1),
            ResBlockXY(128*c, 64*c),
            ResBlockX(64*c),
            nn.GroupNorm(64*c, 64*c),
            nn.SiLU(),
            nn.Conv3d(64*c, 3, (3, 3, 3), padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)

        frames = x.shape[2]
        height = x.shape[3]
        width = x.shape[4]

        x = rearrange(x, 'b c f h w -> b (f h w) c')

        quantized, _, _ = self.vq(x)

        x = rearrange(quantized, 'b (f h w) c -> b c f h w', f=frames, h=height, w=width)

        x = self.decoder(x)

        return x

    def get_n_params(self):
        pp=0
        for p in list(self.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp

class MAGDiscriminator(nn.Module):
    def __init__(self, c) -> None:
        super().__init__()

        self.image_process = nn.Sequential(
            nn.Conv3d(3, 64*c, (3, 3, 3), padding=1),
            nn.LeakyReLU(),
            ResBlockDown(64*c, 128*c),
            ResBlockDown(128*c, 256*c),
            ResBlockDown(256*c, 256*c),
            ResBlockDown(256*c, 256*c),
            ResBlockDown(256*c, 256*c),
            nn.Conv3d(256*c, 256*c, (3, 3, 3), padding=1),
            nn.LeakyReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(256*c, 256*c),
            nn.LeakyReLU(),
            nn.Linear(256*c, 1)
        )

    def forward(self, x):
        x = self.image_process(x)

        x = rearrange(x, 'b c f h w -> b (f h w) c')

        x = self.classifier(x)

        return rearrange(x, 'b c n -> b (c n)').sum(dim=1)

if __name__ == "__main__":
    video = torch.randn((2, 3, 16, 128, 128)) # (batch, channels, frames, height, width)

    model = MAGVQVAE(1)
    print(model.get_n_params())

    output = model(video)
    print(output.shape)

    discr = MAGDiscriminator(1)
    print(discr(output).shape)