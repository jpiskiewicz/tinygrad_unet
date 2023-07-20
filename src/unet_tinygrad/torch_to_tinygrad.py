import sys
import torch
from constants import BATCH_SIZE, OVERLAP, SIZE
from model_torch import Unet3D as Unet3DTorch
from model import Unet3D
import numpy as np
from tinygrad.state import safe_save, safe_load, get_state_dict, load_state_dict
from tinygrad.tensor import Tensor


def copy_weights_block(torch_block, tg_block):
    keys_corresp = {
        ("conv1", 0): 0,
        ("conv1", 1): 1,
        ("conv2", 0): 3,
        ("conv2", 1): 4,
    }
    for tg, tc in keys_corresp.items():
        tg_field = getattr(tg_block, tg[0])[tg[1]]
        if tc not in (1, 4):
            tg_field.weight.assign(torch_block[tc].weight.cpu().numpy())
            tg_field.bias.assign(torch_block[tc].bias.cpu().numpy())

            assert(np.all(tg_field.weight.numpy() == torch_block[tc].weight.cpu().numpy()))
            assert(np.all(tg_field.bias.numpy() == torch_block[tc].bias.cpu().numpy()))

        # try:
        #     tg_field.weight.assign(torch_block[tc].running_mean.cpu().numpy())
        #     tg_field.bias.assign(torch_block[tc].running_var.cpu().numpy())

        #     assert(np.all(tg_field.weight == torch_block[tc].running_mean.cpu().numpy()))
        #     assert(np.all(tg_field.bias == torch_block[tc].running_var.cpu().numpy()))
        # except AttributeError as err:
        #     print(err)



input_weight = sys.argv[1]
output_weight = sys.argv[2]


model_torch = Unet3DTorch()
checkpoint = torch.load(input_weight)
try:
    model_torch.load_state_dict(checkpoint["model_state_dict"])
except RuntimeError:
    print("kkk")
    dmodel = torch.nn.DataParallel(model_torch)
    dmodel.load_state_dict(checkpoint["model_state_dict"])
    model_torch = dmodel.module

# set training flag to false
Tensor.training = False

model_tg = Unet3D()
with torch.no_grad():
    for i in range(1, 5):
        encoder_key = f"encoder{i}"
        print(f"\ncopying {encoder_key}")
        enc_torch = getattr(model_torch, encoder_key)
        enc_tg = getattr(model_tg, encoder_key)
        copy_weights_block(enc_torch, enc_tg)

    for i in range(1, 5):
        decoder_key = f"decoder{i}"
        print(f"\ncopying {decoder_key}")
        enc_torch = getattr(model_torch, decoder_key)
        enc_tg = getattr(model_tg, decoder_key)
        copy_weights_block(enc_torch, enc_tg)

    print(f"copying bottleneck")
    copy_weights_block(model_torch.bottleneck, model_tg.bottleneck)

    print("copying conv")
    model_tg.conv.weight.assign(model_torch.conv.weight.cpu().numpy())
    model_tg.conv.bias.assign(model_torch.conv.bias.cpu().numpy())
    assert np.all(model_tg.conv.weight.numpy() == model_torch.conv.weight.cpu().numpy())
    assert np.all(model_tg.conv.bias.numpy() == model_torch.conv.bias.cpu().numpy())

    for i in range(1, 5):
        upconv_key = f"upconv{i}"
        print(f"\ncopying {upconv_key}")
        enc_torch = getattr(model_torch, upconv_key)
        enc_tg = getattr(model_tg, upconv_key)
        enc_tg.weight.assign(enc_torch.weight.cpu().numpy())
        enc_tg.bias.assign(enc_torch.bias.cpu().numpy())
        assert np.all(enc_tg.weight.numpy() == enc_torch.weight.cpu().numpy())
        assert np.all(enc_tg.bias.numpy() == enc_torch.bias.cpu().numpy())
        print()


state_dict = get_state_dict(model_tg)
safe_save(state_dict, output_weight)