#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import nibabel as nib
from tinygrad import nn
from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_save, get_state_dict, load_state_dict, safe_load
from tinygrad.nn.optim import Adam, Optimizer
from tinygrad.engine.jit import TinyJit
from tqdm import tqdm
from os import path
import random
from model import Unet3D, SIZE
import glob
import json

DATASET = "dataset/MICCAI_BraTS_2019_Data_Training/*GG/*"
DEVICE = "amd"

def choose_files(pattern):
  directories = glob.glob(pattern)
  random.shuffle(directories)
  idx = int(len(directories) * 0.7)
  train = directories[:idx]
  val = directories[idx:]
  with open("validation_files.json", "w") as f: json.dump(val, f)
  return train, val

class BrainMRIDataset:
    def __init__(self, directories, patch_size=SIZE, augment=True):
        self.patch_size = patch_size
        self.augment = augment
        
        self.image_files, self.label_files = self.get_files(directories)
        
        assert len(self.image_files) == len(self.label_files), "Mismatch in image/label pairs"
        print(f"Found {len(self.image_files)} image-label pairs")
        
    def get_files(self, directories): return [[path.join(dirname, Path(dirname).name + "_" + x + ".nii") for dirname in directories] for x in ["t1ce", "seg"]]
    
    def load_nifti(self, filepath):
        """Load and return numpy array from nifti file"""
        img = nib.load(filepath)
        data = img.get_fdata()
        return data.astype(np.float32)
    
    def normalize(self, volume):
        """Normalize volume to [0, 1] range"""
        volume = volume.astype(np.float32)
        min_val = np.percentile(volume, 1)
        max_val = np.percentile(volume, 99)
        volume = np.clip(volume, min_val, max_val)
        if max_val > min_val:
            volume = (volume - min_val) / (max_val - min_val)
        return volume
    
    def extract_random_patch(self, image, label):
        """Extract random patch of size patch_size from image and label"""
        d, h, w = image.shape
        
        # If image is smaller than patch size, pad it
        if d < self.patch_size or h < self.patch_size or w < self.patch_size:
            pad_d = max(0, self.patch_size - d)
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            label = np.pad(label, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            d, h, w = image.shape
        
        # Random crop
        start_d = random.randint(0, d - self.patch_size)
        start_h = random.randint(0, h - self.patch_size)
        start_w = random.randint(0, w - self.patch_size)
        
        image_patch = image[start_d:start_d+self.patch_size,
                           start_h:start_h+self.patch_size,
                           start_w:start_w+self.patch_size]
        label_patch = label[start_d:start_d+self.patch_size,
                           start_h:start_h+self.patch_size,
                           start_w:start_w+self.patch_size]
        
        return image_patch, label_patch
    
    def augment_data(self, image, label):
        """Simple augmentation: random flips"""
        if random.random() > 0.5:
            image = np.flip(image, axis=0).copy()
            label = np.flip(label, axis=0).copy()
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=1).copy()
        if random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
            label = np.flip(label, axis=2).copy()
        return image, label
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image and label
        image = self.load_nifti(self.image_files[idx])
        label = self.load_nifti(self.label_files[idx])
        
        # Normalize image
        image = self.normalize(image)
        
        # Binarize label (in case it's not binary)
        label = (label > 0).astype(np.float32)
        
        # Extract patch
        image_patch, label_patch = self.extract_random_patch(image, label)
        
        # Augment
        if self.augment:
            image_patch, label_patch = self.augment_data(image_patch, label_patch)
        
        # Add channel dimension and batch dimension (1, 1, D, H, W) and two other new dimensions to prepare for concat of
        # image with label and another one before converting the dataset to tensor.
        image_patch = image_patch[*[np.newaxis] * 4]
        label_patch = label_patch[*[np.newaxis] * 4]
        
        return np.concatenate((image_patch, label_patch), axis=1)


def dice_loss(pred: Tensor, target: Tensor, smooth=1e-5) -> Tensor:
    """Dice loss for binary segmentation"""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def dice_coefficient(pred: Tensor, target: Tensor, threshold=0.5, smooth=1e-5) -> Tensor:
    """Dice coefficient metric"""
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice
    
    
def convert_to_tensor(dataset: BrainMRIDataset) -> Tensor:
    CHUNK_SIZE = 20
    dataset_t = Tensor(dataset[0], device=DEVICE).realize()
    for i in range((len(dataset) // CHUNK_SIZE) + 1):
      start_idx = i*CHUNK_SIZE + 1
      count = min(len(dataset) - start_idx, CHUNK_SIZE)
      dataset_t = dataset_t.cat(*[Tensor(dataset[start_idx + x]) for x in range(count)]).realize()
    return dataset_t
    
    
@TinyJit
def tiny_step(idx: int, dataset: Tensor, model: Unet3D, optimizer: Optimizer) -> tuple[Tensor, Tensor]:
    image, label = dataset[idx]
    pred = model(image)
    loss = dice_loss(pred, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    dice = dice_coefficient(pred, label)
    return loss, dice
      

def train_epoch(model, dataset, optimizer):
    """Train for one epoch"""
    indices = list(range(len(dataset)))
    
    # Augument and collect the samples before starting training in this epoch
    # in order for JIT to work.
    epoch_dataset = convert_to_tensor(dataset)
    
    random.shuffle(indices)
    
    total_loss = 0.0
    total_dice = 0.0
    
    with Tensor.train(True):
      for idx in tqdm(indices, desc="Training"):
        loss, dice = tiny_step(idx, epoch_dataset, model, optimizer)
        total_loss += loss.numpy()
        total_dice += dice.numpy()
    
    avg_loss = total_loss / len(dataset)
    avg_dice = total_dice / len(dataset)
    
    return avg_loss, avg_dice


def validate(model: Unet3D, dataset: BrainMRIDataset) -> float:
    """Validate the model"""
    total_dice = 0.0
    
    @TinyJit
    def f(idx: int) -> Tensor:
      image, label = dataset[idx]
      pred = model(image)
      return dice_coefficient(pred, label)
      
    with Tensor.train(False):
      for idx in tqdm(range(len(dataset)), desc="Validating"): total_dice += f(idx).numpy()
      
    return total_dice / len(dataset)
    

def train(
    num_epochs=100,
    learning_rate=1e-4,
    patch_size=SIZE,
    checkpoint_dir='checkpoints',
    resume_from=None
):
    """Main training function"""
    
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Initialize dataset
    print("Loading training and validation data...")
    train_files, val_files = choose_files(DATASET)
    train_dataset = BrainMRIDataset(train_files, patch_size=patch_size, augment=True)
    val_dataset = convert_to_tensor(BrainMRIDataset(val_files, patch_size=patch_size, augment=False))
    
    # Initialize model
    print("Initializing model...")
    model = Unet3D(in_channels=1, out_channels=1, init_features=8)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if resume_from:
        print(f"Loading checkpoint from {resume_from}")
        state_dict = safe_load(resume_from)
        load_state_dict(model, state_dict)
    
    # Move model to device
    model.to(DEVICE)
    
    # Initialize optimizer
    optimizer = Adam(nn.state.get_parameters(model), lr=learning_rate)
    
    # Training loop
    best_dice = 0.0
    
    for epoch in range(start_epoch, num_epochs):
        epoch_msg = f"\nEpoch {epoch+1}/{num_epochs}"
        print(epoch_msg)
        
        # Train
        train_loss, train_dice = train_epoch(model, train_dataset, optimizer)
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        
        # Validate
        val_dice = validate(model, val_dataset)
        val_msg = ", ".join([epoch_msg, f"Epoch Validation Dice: {val_dice:.4f}"])
        print(val_msg)
        with open("eval_scores.txt", "a") as f: f.write(val_msg)
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            save_path = checkpoint_dir / "best_model.safetensors"
            print(f"Saving best model to {save_path}")
            safe_save(get_state_dict(model), save_path)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.safetensors"
            print(f"Saving checkpoint to {save_path}")
            safe_save(get_state_dict(model), save_path)
    
    # Save final model
    final_path = checkpoint_dir / "final_model.safetensors"
    print(f"Saving final model to {final_path}")
    safe_save(get_state_dict(model), final_path)
    
    print("Training complete!")


if __name__ == '__main__':
    # Example usage
    train(
        num_epochs=50,
        learning_rate=1e-4,
        patch_size=SIZE,
        checkpoint_dir='checkpoints',
        resume_from=None
    )