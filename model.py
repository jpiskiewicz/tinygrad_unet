from pathlib import Path
from tinygrad import nn
from tinygrad.tensor import Tensor
import numpy as np
from tinygrad.state import safe_save, safe_load, get_state_dict, load_state_dict

SIZE = 48


class BatchNorm3d:
  def __init__(self, sz, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
    self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum

    if affine: self.weight, self.bias = Tensor.ones(sz), Tensor.zeros(sz)
    else: self.weight, self.bias = None, None

    self.running_mean, self.running_var = Tensor.zeros(sz, requires_grad=False), Tensor.ones(sz, requires_grad=False)
    self.num_batches_tracked = Tensor.zeros(1, requires_grad=False)

  def __call__(self, x:Tensor):
    # NOTE: this can be precomputed for static inference. we expand it here so it fuses
    batch_invstd = self.running_var.reshape(1, -1, 1, 1, 1).expand(x.shape).add(self.eps).rsqrt()
    bn_init = (x - self.running_mean.reshape(1, -1, 1, 1, 1).expand(x.shape)) * batch_invstd
    return self.weight.reshape(1, -1, 1, 1, 1).expand(x.shape) * bn_init + self.bias.reshape(1, -1, 1, 1, 1).expand(x.shape)


class DownsampleBlock:
    def __init__(self, in_channels, features, padding=1, kernel_size=5, stride=1, name="block"):
        self.conv1 = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=(kernel_size, kernel_size, kernel_size),
                padding=(padding, padding, padding, padding, padding, padding),
                stride=(stride, stride, stride),
                bias=True,
            ),
            BatchNorm3d(features),
            Tensor.relu,
        ]

        self.conv2 = [
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=(kernel_size, kernel_size, kernel_size),
                padding=(padding, padding, padding, padding, padding, padding),
                stride=(stride, stride, stride),
                bias=True,
            ),
            BatchNorm3d(features),
            Tensor.relu,
        ]

    def __call__(self, x):
        return x.sequential(self.conv1).sequential(self.conv2)


class UpsampleBlock:
    def __init__(self, in_channels, features, padding=1, kernel_size=5, stride=1, name="block"):
        self.conv1 = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=(kernel_size, kernel_size, kernel_size),
                padding=(padding, padding, padding, padding, padding, padding),
                stride=(stride, stride, stride),
                bias=True,
            ),
            BatchNorm3d(features),
            Tensor.relu,
        ]

        self.conv2 = [
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=(kernel_size, kernel_size, kernel_size),
                padding=(padding, padding, padding, padding, padding, padding),
                stride=(stride, stride, stride),
                bias=True,
            ),
            BatchNorm3d(features),
            Tensor.relu,
        ]

    def __call__(self, x):
        return x.sequential(self.conv1).sequential(self.conv2)



class Unet3D:
    def __init__(self, in_channels=1, out_channels=1, init_features=8):
        features = init_features
        self.encoder1 = DownsampleBlock(
            in_channels, features=features, padding=2, name="enc1"
        )

        self.encoder2 = DownsampleBlock(
            features, features=features * 2, padding=2, name="enc1"
        )

        self.encoder3 = DownsampleBlock(
            features * 2, features=features * 4, padding=2, name="enc1"
        )

        self.encoder4 = DownsampleBlock(
            features * 4 , features=features * 8, padding=2, name="enc1"
        )

        self.bottleneck = DownsampleBlock(
            features * 8, features=features * 16, padding=2, name="bottleneck"
        )

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=(4, 4, 4), stride=2, padding=1
        )

        self.decoder4 = UpsampleBlock(
            features * 16, features=features * 8, padding=2, name="dec4"
        )

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=(4, 4, 4), stride=2, padding=1
        )

        self.decoder3 = UpsampleBlock(
            features * 8, features=features * 4, padding=2, name="dec4"
        )

        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=(4, 4, 4), stride=2, padding=1
        )

        self.decoder2 = UpsampleBlock(
            features * 4, features=features * 2, padding=2, name="dec4"
        )

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=(4, 4, 4), stride=2, padding=1
        )

        self.decoder1 = UpsampleBlock(
            features * 2, features=features, padding=2, name="dec4"
        )

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=(1, 1, 1)
        )

    def __call__(self, x):
        enc1 = self.encoder1(x)
        o0 = enc1.max_pool2d(kernel_size=(2, 2, 2), stride=2)
        enc2 = self.encoder2(o0)
        o1 = enc2.max_pool2d(kernel_size=(2, 2, 2), stride=2)
        enc3 = self.encoder3(o1)
        o2 = enc3.max_pool2d(kernel_size=(2, 2, 2), stride=2)
        enc4 = self.encoder4(o2)
        o3 = enc4.max_pool2d(kernel_size=(2, 2, 2), stride=2)

        bottleneck = self.bottleneck(o3)

        upconv4 = self.upconv4(bottleneck)
        dec4 = upconv4.cat(enc4, dim=1)
        dec4 = self.decoder4(dec4)

        upconv3 = self.upconv3(dec4)
        dec3 = upconv3.cat(enc3, dim=1)
        dec3 = self.decoder3(dec3)

        upconv2 = self.upconv2(dec3)
        dec2 = upconv2.cat(enc2, dim=1)
        dec2 = self.decoder2(dec2)

        upconv1 = self.upconv1(dec2)
        dec1 = upconv1.cat(enc1, dim=1)
        dec1 = self.decoder1(dec1)

        conv = self.conv(dec1)

        sigmoid = conv.sigmoid()

        return sigmoid


if __name__ == '__main__':
    print('here')
    Tensor.training = False
    arr = np.random.randn(1, 1, SIZE, SIZE, SIZE)
    arr = Tensor(arr.astype(np.float32))
    m = Unet3D()
    state_dict = safe_load("brain_mri_t1.safetensors")
    load_state_dict(m, state_dict)
    m(arr)
