## 1. Problem Understanding
- **Task type:** Binary Image Segmentation (identifying salt deposits in seismic imagery).
- **Evaluation metric:** Mean Average Precision (mAP) at different Intersection over Union (IoU) thresholds, though optimized proxy metrics include BCE, Dice, and Lovasz-Hinge loss.
- **Key challenges:** Small image size (101x101), high noise, optimization of discrete IoU metric, and leveraging unlabeled/test data effectively via pseudolabeling. 

## 2. Data Pipeline (Code-Oriented)
- **`load_data()`**: 
  - Read `train.csv`, `depths.csv`, and `sample_submission.csv`. 
  - Merge train data with depth information (crucial for stratification).
  - Load images and masks as NumPy arrays (grayscale converted to RGB for pretrained models).
- **`preprocess()`**: 
  - Apply spatial transformations: Pad (101x101 $\to$ 128x128 or 256x256) or Resize + Pad (101x101 $\to$ 192x192 $\to$ 224x224).
  - Implement augmentations via Albumentations library: `HorizontalFlip(p=0.5)`, `RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3)`, and `ShiftScaleRotate(shift_limit=0.1625, scale_limit=0.6, rotate_limit=0, p=0.7)`.
  - Normalize using ImageNet statistics.
- **`split_folds()`**: 
  - Discretize the continuous `depth` variable into 5-10 bins.
  - Apply `StratifiedKFold(n_splits=5)` using the depth bins as the target label to ensure depth distribution is uniform across folds.

## 3. Model Design
- **`build_model(encoder_name, input_shape)`**:
  - Implement a U-Net style architecture.
  - **Encoders:** Load `ResNeXt50` or `ResNet34` pretrained on ImageNet. Modify the first max-pooling layer (remove or bypass it to preserve spatial resolution for small inputs).
  - **Center Block:** Implement either standard convolutions, Feature Pyramid Attention (FPA), or Global Convolutional Network (GCN).
  - **Decoders:** Use `conv3x3`, Batch Normalization, and Upsampling/Transposed Convolutions. 
  - **Attention/Refinement:** Integrate Spatial and Channel Squeeze & Excitation (scSE) blocks in the decoder to re-weight feature maps. Option to add Hypercolumns (concatenating multiscale features before the final classifier).

## 4. Training Strategy
- **`train_one_fold(fold_idx, train_loader, val_loader, stage)`**:
  - **Stage 1 (Supervised Base):**
    - Optimizer: RMSprop (batch size 24-32).
    - Phase 1: Train with `BCE + Dice Loss`. Use `ReduceLROnPlateau` starting at 1e-4.
    - Phase 2: Switch to `Lovasz-Hinge Loss`. `ReduceLROnPlateau` starting at 5e-5.
    - Phase 3: Continue `Lovasz Loss` with Cosine Annealing (4 snapshot cycles, 80 epochs each, max LR 1e-4).
  - **Stage 2 & 3 (Pseudolabel Pretraining & Finetuning):**
    - Optimizer: SGD.
    - Pretrain phase: Train on combined train + confident pseudolabels (150 epochs, 3 cycles cosine annealing, LR 0.01 $\to$ 0.001).
    - Finetune phase: Train *only* on ground-truth train data (4 snapshots cosine annealing, 50 epochs each, LR 0.01 $\to$ 0.001).

## 5. Validation Strategy
- **`validate(model, val_loader)`**:
  - Run inference on the validation fold without TTA during standard epochs to save time.
  - Compute standard IoU and Lovasz metrics.
  - Save Out-Of-Fold (OOF) predictions.
  - Select the best model weights based on the validation Lovasz/IoU score.
- **`find_optimal_threshold(oof_preds, y_true)`**:
  - Iterate over thresholds (e.g., 0.3 to 0.7) to maximize the competition specific mAP metric.

## 6. Inference Pipeline
- **`predict(model, test_loader)`**:
  - Implement Test-Time Augmentation (TTA). For this, use standard prediction and horizontally flipped prediction: `pred = (model(img) + fliplr(model(fliplr(img)))) / 2.0`.
- **`generate_pseudolabels(test_preds)`**:
  - Filter predictions to find confident pixels (`prob < 0.2` or `prob > 0.8`). If a test image has a high percentage of confident pixels, assign it as a pseudolabel for subsequent training stages.
- **`post_process(preds, depth_info)`**:
  - Remove masks below a certain minimum pixel size.
  - (Optional) Implement Jigsaw mosaic logic: force zero masks for specific vertical cascade rules based on image spatial metadata (Note: Proceed with caution as this overfits public LB).

## 7. Key Tricks (ACTIONABLE)
- **If initial training stalls $\to$** Switch loss functions. Start with smooth BCE+Dice for convergence, then switch to Lovasz to directly optimize the discrete IoU-like metric.
- **If you have high-quality OOF models $\to$** Generate pseudolabels. Take the test set, predict using a Stage 1 ensemble, threshold at 0.2/0.8, and add these hard-confident targets to the Stage 2 training set.
- **If input is 101x101 $\to$** Do not use standard 5-stage U-Net directly. Pad to 128x128 (to allow clean divisibility by 2 for downsampling) or resize to 192x192 and pad to 224x224.
- **If features need localized context $\to$** Inject scSE blocks into every decoder layer. 

## 8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)

```python
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from sklearn.model_selection import StratifiedKFold

# --- CONFIGURATION ---
class Config:
    SEED = 42
    N_FOLDS = 5
    BATCH_SIZE = 32
    # Define other hyperparameters

def seed_everything(seed):
    # Set seeds for torch, numpy, os, etc.
    pass

# --- DATA PIPELINE ---
def load_data():
    # Load train.csv, depths.csv, sample_submission.csv
    # Merge depths into train, return dataframes
    pass

class SaltDataset(Dataset):
    def __init__(self, df, transforms=None, is_test=False):
        # Image loading and transform application
        pass
    def __len__(self): pass
    def __getitem__(self, idx): pass

def get_transforms(phase):
    # Return Albumentations augmentations for train/val/test
    pass

def create_folds(df):
    # Discretize depth and use StratifiedKFold
    pass

# --- MODEL DESIGN ---
class SCSEBlock(nn.Module):
    # Spatial and Channel Squeeze & Excitation
    pass

class ResNeXtUNet(nn.Module):
    # Encoder: ResNeXt/ResNet
    # Decoder: U-Net blocks + SCSE
    pass

def build_model(config):
    # Initialize and return model
    pass

# --- LOSSES ---
def bce_dice_loss(pred, target): pass
def lovasz_hinge_loss(pred, target): pass

# --- TRAINING STRATEGY ---
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    # Forward pass, loss, backward, step
    pass

def validate(model, dataloader, criterion, device):
    # Validation loop without gradients, calculate IoU
    pass

def train_one_fold(fold, train_df, val_df, pseudo_df=None):
    # 1. Initialize dataset & loaders (include pseudo_df if stage > 1)
    # 2. Build model, optimizer (RMSprop/SGD), scheduler (Cosine/Plateau)
    # 3. Phase 1 loop: BCE+Dice
    # 4. Phase 2 loop: Lovasz
    # 5. Save best weights
    # Return model and OOF predictions
    pass

# --- INFERENCE & PSEUDOLABELING ---
def generate_pseudolabels(test_preds, test_df, conf_threshold_low=0.2, conf_threshold_high=0.8):
    # Extract confident predictions as new ground truths
    pass

def inference(models, test_loader, device):
    # Loop through models, apply horizontal flip TTA, average preds
    pass

def post_process(preds):
    # Remove tiny masks, apply mosaic heuristics if desired
    pass

def save_submission(preds, test_df):
    # RLE encode masks and save to csv
    pass

# --- MAIN EXECUTION ---
def main():
    seed_everything(Config.SEED)
    train_df, test_df = load_data()
    train_df = create_folds(train_df)
    
    # --- STAGE 1 ---
    models_stage1 = []
    oof_stage1 = np.zeros(len(train_df))
    for fold in range(Config.N_FOLDS):
        model, oof = train_one_fold(fold, train_df[train_df.fold != fold], train_df[train_df.fold == fold])
        models_stage1.append(model)
        
    # --- STAGE 2 (Pseudolabeling) ---
    test_preds_s1 = inference(models_stage1, test_loader, device)
    pseudo_df = generate_pseudolabels(test_preds_s1, test_df)
    
    models_stage2 = []
    for fold in range(Config.N_FOLDS):
        # Train using pseudo_df for pretraining, then finetune on fold train_df
        model, oof = train_one_fold(fold, train_df[train_df.fold != fold], train_df[train_df.fold == fold], pseudo_df)
        models_stage2.append(model)

    # --- INFERENCE ---
    final_preds = inference(models_stage2, test_loader, device)
    final_preds = post_process(final_preds)
    save_submission(final_preds, test_df)

if __name__ == "__main__":
    main()
```

## 9. Strategy Priority (IMPORTANT)
1. **Most impactful techniques:** Multi-stage training with confident pseudolabeling (Stage 2/3), Lovasz-Hinge Loss for direct metric optimization, and Depth-stratified K-fold validation.
2. **Secondary improvements:** Custom decoder modules (scSE blocks, Feature Pyramid Attention), specific spatial padding (ensuring spatial dimensions divide cleanly in the U-Net), and Cosine Annealing snapshots for robust ensembles.
3. **Minor tricks:** Horizontal flip TTA, Albumentations (ShiftScaleRotate), mosaic-based postprocessing heuristics.