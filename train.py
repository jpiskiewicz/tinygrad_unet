#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import nibabel as nib
from tinygrad import nn
from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_save, get_state_dict, load_state_dict, safe_load
from tinygrad.nn.optim import Adam
from tqdm import tqdm
from os import path
import random
from model import Unet3D, SIZE
import glob
import json

DATASET = "brain-tumor-segmentation-brats-2019/MICCAI_BraTS_2019_Data_Training/*GG/*"
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
        
        # Add channel dimension: (1, D, H, W)
        image_patch = np.expand_dims(image_patch, axis=0)
        label_patch = np.expand_dims(label_patch, axis=0)
        
        return image_patch, label_patch


def dice_loss(pred, target, smooth=1e-5):
    """Dice loss for binary segmentation"""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def dice_coefficient(pred, target, threshold=0.5, smooth=1e-5):
    """Dice coefficient metric"""
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def train_epoch(model, dataset, optimizer):
    """Train for one epoch"""
    total_loss = 0.0
    total_dice = 0.0
    
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    with Tensor.train(True):
      for idx in tqdm(indices, desc="Training"):
          image, label = dataset[idx]
          
          # Convert to tensors and add batch dimension
          image_tensor = Tensor(image[np.newaxis, ...], device=DEVICE, requires_grad=False)
          label_tensor = Tensor(label[np.newaxis, ...], device=DEVICE, requires_grad=False)
          
          # Forward pass
          pred = model(image_tensor)
          
          # Calculate loss
          loss = dice_loss(pred, label_tensor)
          
          # Backward pass
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
          # Metrics
          dice = dice_coefficient(pred, label_tensor)
          
          total_loss += loss.numpy()
          total_dice += dice.numpy()
    
    avg_loss = total_loss / len(dataset)
    avg_dice = total_dice / len(dataset)
    
    return avg_loss, avg_dice


def validate(model, dataset):
    """Validate the model"""
    Tensor.training = False
    total_dice = 0.0
    
    for idx in tqdm(range(len(dataset)), desc="Validating"):
        image, label = dataset[idx]
        
        # Convert to tensors and add batch dimension
        image_tensor = Tensor(image[np.newaxis, ...], device=DEVICE, requires_grad=False)
        label_tensor = Tensor(label[np.newaxis, ...], device=DEVICE, requires_grad=False)
        
        # Forward pass
        pred = model(image_tensor)
        
        # Calculate dice
        dice = dice_coefficient(pred, label_tensor)
        total_dice += dice.numpy()
    
    avg_dice = total_dice / len(dataset)
    return avg_dice


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
    val_dataset = BrainMRIDataset(val_files, patch_size=patch_size, augment=False)
    
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
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_dice = train_epoch(model, train_dataset, optimizer)
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        
        # Validate
        if val_dataset:
            val_dice = validate(model, val_dataset)
            print(f"Validation Dice: {val_dice:.4f}")
            
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
        num_epochs=1000,
        learning_rate=1e-5,
        patch_size=SIZE,
        checkpoint_dir='checkpoints',
        resume_from=None
    )