# MAGVIT-pytorch
Implementation of MAGVIT in pytorch

⚠️ WIP stage, may not work perfectly!

Original paper: https://arxiv.org/pdf/2212.05199.pdf

## Usage

```python
video = torch.randn((2, 3, 16, 128, 128)) # (batch, channels, frames, height, width)

model = MAGVQVAE(1)
print(model.get_n_params())

output = model(video)
print(output.shape)

discr = MAGDiscriminator(1)
print(discr(output).shape)
```

## Citations

```bibtex
@misc{yu2022magvit,
    title   = {MAGVIT: Masked Generative Video Transformer},
    author  = {Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G. Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, Lu Jiang},
    year    = {2022},
    url     = {https://arxiv.org/abs/2212.05199}
}
```
