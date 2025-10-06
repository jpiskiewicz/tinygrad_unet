#!/usr/bin/env python3

import argparse
import itertools
import pathlib
import typing

import nibabel as nb
import numpy as np
from tinygrad import Device, dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.state import (
    load_state_dict,
    safe_load,
)
from tinygrad.tensor import Tensor
from tqdm import tqdm
from vtk.util import numpy_support

from constants import BATCH_SIZE, OVERLAP, SIZE
from model import Unet3D

# # https://github.com/nipy/nibabel/issues/626#issuecomment-386338532
# nb.Nifti1Header.quaternion_threshold = -1e-06

# set training flag to false
Tensor.training = False


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-w",
    "--weights",
    default="brain_mri_t1.safetensors",
    type=pathlib.Path,
    metavar="path",
    help="Weight path",
    dest="weights",
)
parser.add_argument(
    "-i",
    "--input",
    type=pathlib.Path,
    metavar="path",
    help="Nifti input file",
    dest="input_file",
    required=True,
)
parser.add_argument(
    "-o",
    "--output",
    default="output.nii",
    type=pathlib.Path,
    metavar="path",
    help="VTI output file",
    dest="output_file",
)
parser.add_argument(
    "-d",
    "--device",
    default="amd",
    type=str,
    help="Which device to use: cpu, cuda, xpu, mkldnn, opengl, opencl, ideep, hip, msnpu, xla, vulkan",
    dest="device",
)
parser.add_argument("--ww", default=None, type=int, dest="window_width")
parser.add_argument("--wl", default=None, type=int, dest="window_level")
parser.add_argument(
    "-b", "--batch_size", default=BATCH_SIZE, type=int, dest="batch_size"
)


def image_normalize(
    image: np.ndarray,
    min_: float = 0.0,
    max_: float = 1.0,
    output_dtype: np.dtype = np.int16,
) -> np.ndarray:
    output = np.empty(shape=image.shape, dtype=output_dtype)
    imin, imax = image.min(), image.max()
    output[:] = (image - imin) * ((max_ - min_) / (imax - imin)) + min_
    return output


def get_LUT_value_255(image: np.ndarray, window: int, level: int) -> np.ndarray:
    shape = image.shape
    data_ = image.ravel()
    image = np.piecewise(
        data_,
        [
            data_ <= (level - 0.5 - (window - 1) / 2),
            data_ > (level - 0.5 + (window - 1) / 2),
        ],
        [0, 255, lambda data_: ((data_ - (level - 0.5)) / (window - 1) + 0.5) * (255)],
    )
    image.shape = shape
    return image


def gen_patches(
    image: np.ndarray, patch_size: int, overlap: int, batch_size: int = BATCH_SIZE
) -> typing.Iterator[typing.Tuple[float, np.ndarray, typing.Iterable]]:
    sz, sy, sx = image.shape
    i_cuts = list(
        itertools.product(
            range(0, sz - patch_size, patch_size - overlap),
            range(0, sy - patch_size, patch_size - overlap),
            range(0, sx - patch_size, patch_size - overlap),
        )
    )
    patches = []
    indexes = []
    for idx, (iz, iy, ix) in enumerate(i_cuts):
        ez = iz + patch_size
        ey = iy + patch_size
        ex = ix + patch_size
        patch = image[iz:ez, iy:ey, ix:ex]
        patches.append(patch)
        indexes.append(((iz, ez), (iy, ey), (ix, ex)))
        if len(patches) == batch_size:
            yield (idx + 1.0) / len(i_cuts), np.asarray(patches), indexes
            patches = []
            indexes = []
    if patches:
        yield 1.0, np.asarray(patches), indexes


def pad_image(image: np.ndarray, patch_size: int = SIZE) -> np.ndarray:
    sz, sy, sx = image.shape
    pad_z = int(np.ceil(sz / patch_size) * patch_size) - sz + OVERLAP
    pad_y = int(np.ceil(sy / patch_size) * patch_size) - sy + OVERLAP
    pad_x = int(np.ceil(sx / patch_size) * patch_size) - sx + OVERLAP
    padded_image = np.pad(image, ((0, pad_z), (0, pad_y), (0, pad_x)))
    return padded_image


def brain_segment(
    image: np.ndarray,
    model: Unet3D,
    dev,
    mean: float,
    std: float,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    dz, dy, dx = image.shape
    image = image_normalize(image, 0.0, 1.0, output_dtype=np.float32)
    padded_image = pad_image(image, SIZE)
    padded_image = (padded_image - mean) / std
    probability_array = np.zeros_like(padded_image, dtype=np.float32)
    sums = np.zeros_like(padded_image)
    pbar = tqdm()
    # segmenting by patches
    for completion, patches, indexes in gen_patches(
        padded_image, SIZE, OVERLAP, batch_size
    ):
        pred = model(Tensor(patches.reshape(-1, 1, SIZE, SIZE, SIZE), dtype=dtypes.float32, device=dev, requires_grad=False)).numpy()
        for i, ((iz, ez), (iy, ey), (ix, ex)) in enumerate(indexes):
            probability_array[iz:ez, iy:ey, ix:ex] += pred[i, 0]
            sums[iz:ez, iy:ey, ix:ex] += 1
        pbar.set_postfix(completion=completion * 100)
        pbar.update()
    pbar.close()
    probability_array[:dz, :dy, :dx] /= sums[:dz, :dy, :dx]
    return np.array(probability_array[:dz, :dy, :dx])


def do_jit(net):
    @TinyJit
    def jit(x):
        return net(x).realize()

    return jit

    
def image_save_nifti(image: np.ndarray, filename: str, reference_nii: nb.Nifti1Image):
    """Save image as NIfTI file, preserving header information from reference."""
    # Create new NIfTI image with the same affine and header as the input
    nii_img = nb.Nifti1Image(image, reference_nii.affine, reference_nii.header)
    
    # Update the data type in the header to match the output
    nii_img.set_data_dtype(image.dtype)
    
    # Save the file
    nb.save(nii_img, filename)
    print(f"Saved output to: {filename}")


def main():
    args, _ = parser.parse_known_args()
    input_file = args.input_file
    weights_file = args.weights
    output_file = args.output_file
    device = args.device
    if not device:
        device = Device.DEFAULT
    nii_data = nb.load(str(input_file))
    image = nii_data.get_fdata()
    mean = 0.0
    std = 1.0
    model = Unet3D()
    state_dict = safe_load(weights_file)
    model.to(device)
    load_state_dict(model, state_dict)
    print(
        f"mean={mean}, std={std}, {image.min()=}, {image.max()=}, {args.window_width=}, {args.window_level=}"
    )

    if args.window_width is not None and args.window_level is not None:
        image = get_LUT_value_255(image, args.window_width, args.window_level)
        print("ww wl", image.min(), image.max())

    # probability_array = brain_segment(image, model, dev, 0.0, 1.0)
    model_jit = do_jit(model)
    probability_array = brain_segment(image, model_jit, device, mean, std, args.batch_size)
    image_save_nifti(probability_array, str(output_file), nii_data)


if __name__ == "__main__":
    main()
