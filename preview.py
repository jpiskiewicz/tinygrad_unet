#!/usr/bin/env python3

# Based on:
# https://neuraldatascience.io/8-mri/nifti.html
 
import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np
from os import path
from pathlib import Path
import sys

CHANNEL_NAMES = ["t1", "t1ce", "t2", "flair", "seg"]
dirname, slice = sys.argv[1], int(sys.argv[2])
filenames = [*[path.join(dirname, Path(dirname).name + "_" + x + ".nii") for x in CHANNEL_NAMES], "output.nii"]
data = [np.rot90(nib.load(x).get_fdata(), axes=(1, 2)) for x in filenames]
fig, axes = plt.subplots(2, 4, figsize=(16, 3))
fig.delaxes(axes[1, 2])
fig.delaxes(axes[1, 3])
fig.suptitle(dirname)
for i, (ax, channel_name) in enumerate(zip(axes.flatten(), [*CHANNEL_NAMES, "out"])):
  channel_data = data[i]
  im = ax.imshow(channel_data[slice])
  ax.set_title(f'{channel_name} - Slice {slice}')
  ax.axis('off')
plt.tight_layout() 
plt.show()
