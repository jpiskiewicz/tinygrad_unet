import sys
import torch
from constants import BATCH_SIZE, OVERLAP, SIZE
from model_torch import Unet3D as Unet3DTorch
from model import Unet3D
import numpy as np
from tinygrad.state import safe_save, safe_load, get_state_dict, load_state_dict


def compare_weights_block(torch_block, tg_block):
    keys_corresp = {
        ("conv1", 0): 0,
        ("conv1", 1): 1,
        ("conv2", 0): 3,
        ("conv2", 1): 4,
    }
    for tg, tc in keys_corresp.items():
        tg_field = getattr(tg_block, tg[0])[tg[1]]

        assert np.all(tg_field.weight.numpy() == torch_block[tc].weight.cpu().numpy())
        assert np.all(tg_field.bias.numpy() == torch_block[tc].bias.cpu().numpy())

        try:
            assert(np.all(tg_field.running_mean == torch_block[tc].running_mean.cpu().numpy()))
            assert(np.all(tg_field.running_var == torch_block[tc].running_var.cpu().numpy()))
        except AttributeError as err:
            print(err)


torch_weight = sys.argv[1]
tinygrad_weight = sys.argv[2]


model_torch = Unet3DTorch()
checkpoint = torch.load(torch_weight)
try:
    model_torch.load_state_dict(checkpoint["model_state_dict"])
except RuntimeError:
    print("kkk")
    dmodel = torch.nn.DataParallel(model_torch)
    dmodel.load_state_dict(checkpoint["model_state_dict"])
    model_torch = dmodel.module

model_tg = Unet3D()
state_dict = safe_load(tinygrad_weight)
load_state_dict(model_tg, state_dict)
with torch.no_grad():
    for i in range(1, 5):
        encoder_key = f"encoder{i}"
        print(f"\ncomparing {encoder_key}")
        enc_torch = getattr(model_torch, encoder_key)
        enc_tg = getattr(model_tg, encoder_key)
        compare_weights_block(enc_torch, enc_tg)

    for i in range(1, 5):
        decoder_key = f"decoder{i}"
        print(f"\ncomparing {decoder_key}")
        enc_torch = getattr(model_torch, decoder_key)
        enc_tg = getattr(model_tg, decoder_key)
        compare_weights_block(enc_torch, enc_tg)

    print(f"comparing bottleneck")
    compare_weights_block(model_torch.bottleneck, model_tg.bottleneck)

    print("comparing conv")
    assert np.all(model_tg.conv.weight.numpy() == model_torch.conv.weight.cpu().numpy())
    assert np.all(model_tg.conv.bias.numpy() == model_torch.conv.bias.cpu().numpy())

    for i in range(1, 5):
        upconv_key = f"upconv{i}"
        print(f"\ncomparing {upconv_key}")
        enc_torch = getattr(model_torch, upconv_key)
        enc_tg = getattr(model_tg, upconv_key)
        assert np.all(enc_tg.weight.numpy() == enc_torch.weight.cpu().numpy())
        assert np.all(enc_tg.bias.numpy() == enc_torch.bias.cpu().numpy())
        print()