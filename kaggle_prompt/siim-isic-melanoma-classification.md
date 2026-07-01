## 1. Problem Understanding
* **Task Type:** Computer Vision (Image Classification) coupled with Tabular Data (Metadata).
* **Evaluation Metric:** ROC-AUC.
* **Key Challenges:** Extreme class imbalance (very small proportion of positive melanoma samples), leading to massive variance between cross-validation (CV) and public Leaderboard (LB) scores. A robust validation setup and heavy ensembling are mandatory to survive private LB shake-ups.

## 2. Data Pipeline (Code-Oriented)

* **`load_data()`**:
    * Read train and test CSVs.
    * Concatenate datasets from 2018, 2019, and 2020 to increase the total number of samples and positive targets, ensuring a stable CV.
* **`preprocess()`**:
    * Map the 2020 diagnoses to the 2019 9-class formulation to unify the datasets. 
    * Implement mapping dictionary: `{'seborrheic keratosis': 'BKL', 'lichenoid keratosis': 'BKL', 'solar lentigo': 'BKL', 'lentigo NOS': 'BKL', 'cafe-au-lait macule': 'unknown', 'atypical melanocytic proliferation': 'unknown', 'nevus': 'NV', 'melanoma': 'MEL'}`.
    * Convert the mapped string labels into integers (0-8). Store the integer index of `MEL` for later inference.
* **`feature_engineering()`**:
    * Extract and normalize the 14 metadata features (e.g., age, sex, anatom_site_general_challenge, etc.). Fill missing tabular values with dataset means or modes.
* **`split_folds()`**:
    * Implement triple-stratified K-Fold (stratifying by target, patient ID, and image count per patient) to prevent data leakage. Use `n_splits=5`.
* **`get_transforms()`**:
    * Use Albumentations for heavy data augmentation on training data: Transpose, VerticalFlip, HorizontalFlip (p=0.5), RandomBrightness/Contrast (limit 0.2, p=0.75), blurring/noise variants (p=0.7), optical/grid distortions (p=0.7), CLAHE (p=0.7), HueSaturationValue (p=0.5), ShiftScaleRotate (p=0.85), Cutout (1 hole, 37.5% size, p=0.7), and Normalize.
    * Validation simply resizes and normalizes. Input sizes iterate between 384 and 896 depending on the model variation.

## 3. Model Design
* **`build_model()`**:
    * Framework: PyTorch (chosen over TF/TPU for better CV correlation and experimental flexibility).
    * Backbones: EfficientNet (B3 through B7), SE-ResNeXt101, and ResNeSt101. Load pretrained weights.
    * Architecture: Replace the final classification head. If metadata is included, concatenate the global average pooled CNN features with the 14 tabular features. Pass this concatenated vector through a Dropout layer and a final Linear layer.
    * Output: The final Linear layer must output 9 logits (matching the unified 9-class diagnosis targets).

## 4. Training Strategy
* **`train_one_fold()`**:
    * **Loss Function:** Standard Multiclass Cross-Entropy Loss (`nn.CrossEntropyLoss()`). This is critical—using 9-class CE instead of Binary Cross-Entropy (BCE) on the single melanoma target boosts performance by ~0.01 AUC.
    * **Optimizer / Params:** AdamW optimizer with Cosine Annealing Learning Rate scheduler.
    * **Tricks:** Implement PyTorch Native Automatic Mixed Precision (AMP) using `torch.cuda.amp.autocast()` and `GradScaler()` to speed up training for large image sizes (up to 896x896) and fit them into GPU memory.

## 5. Validation Strategy
* **Cross-Validation Logic:** Validate on the 5 folds. 
* **OOF Generation:** Track Out-Of-Fold (OOF) predictions for the `MEL` class by applying softmax to the 9 output logits and slicing the probability of the `MEL` index.
* **Metric Tracking:** Calculate AUC twice: once on the entire validation set (`cv_all`) and once isolating only the 2020 data subset (`cv_2020`). Base model selection on `cv_all` for maximum stability.

## 6. Inference Pipeline
* **`predict()`**:
    * Load the best weights for each fold.
    * **TTA:** Apply Test Time Augmentation 8x (combinations of horizontal flips, vertical flips, and original image) to stabilize predictions.
    * Extract the `MEL` class probability via `F.softmax(logits, dim=1)[:, MEL_INDEX]`. Average the TTA predictions for the image.
* **`post_process()`**:
    * Before averaging predictions across different folds or different model architectures, convert probabilities to percentiles to enforce uniform distributions. 
    * Apply `df['pred'] = df['pred'].rank(pct=True)` on the test predictions of each model before creating the final arithmetic mean ensemble.

## 7. Key Tricks (ACTIONABLE)
* **If defining the target space →** Do NOT use 0/1 binary targets. Map all diagnoses to a 9-class system and use `CrossEntropyLoss`.
* **If setting up validation →** Include 2018 and 2019 data to boost positive sample size, but isolate metrics to monitor both `cv_all` and `cv_2020`.
* **If ensembling →** Do NOT average raw probabilities directly. Rank-transform (`pct=True`) every model's predictions before blending.

## 8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)

```python
import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import autocast, GradScaler
import timm

# --- Configuration ---
class Config:
    seed = 42
    epochs = 10
    batch_size = 16
    image_size = 512
    n_splits = 5
    lr = 1e-4
    mel_idx = 7 # Adjust based on label encoding mapping

def seed_everything(seed):
    # Set seeds for os, numpy, and torch to ensure reproducibility

def get_transforms(is_train):
    # Return heavy Albumentations compose for train, standard for val

class MelanomaDataset(Dataset):
    # Inherit Dataset, load image via cv2, apply transforms
    # Return image tensor, metadata tensor, and target label (0-8)

def load_data():
    # Load 2018, 2019, 2020 CSVs. Concat.
    # Map 2020 targets to 2019 target space (9 classes).
    # Return train_df and test_df

def feature_engineering(df):
    # Impute missing values in 14 metadata columns.
    # Scale/normalize continuous features, encode categorical features.
    # Return dataframe with processed tabular features

def create_folds(df):
    # Apply triple stratified KFold (stratify by target and patient_id)
    # Assign 'fold' column to dataframe

class MelanomaModel(nn.Module):
    # Init CNN backbone (timm.create_model)
    # Define linear layer for tabular data
    # Concat pooled CNN features + Tabular features
    # Final linear layer with out_features=9

def train_one_fold(fold, train_df, val_df):
    # Setup DataLoaders, Model, Loss (CrossEntropy), Optimizer (AdamW), Scaler (AMP)
    # Loop epochs:
        # train_one_epoch
        # validate -> track cv_all and cv_2020
    # Save best model based on cv_all OOF
    # Return OOF predictions for this fold

def validate(model, val_loader):
    # Run model in eval mode
    # Calculate F.softmax()[:, MEL_INDEX]
    # Return ROC-AUC score

def inference(models, test_df):
    # For each model, run test dataloader 8 times with TTA
    # Average TTA predictions
    # Rank convert probabilities: pd.Series(preds).rank(pct=True)
    # Average across all models/folds

def main():
    seed_everything(Config.seed)
    train_df, test_df = load_data()
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    train_df = create_folds(train_df)
    
    models = []
    oof_preds = np.zeros(len(train_df))
    
    for fold in range(Config.n_splits):
        trn_idx = train_df[train_df['fold'] != fold].index
        val_idx = train_df[train_df['fold'] == fold].index
        oof = train_one_fold(fold, train_df.iloc[trn_idx], train_df.iloc[val_idx])
        oof_preds[val_idx] = oof
        models.append(f'model_fold_{fold}.pth')
        
    preds = inference(models, test_df)
    
    submission = pd.DataFrame({'image_name': test_df['image_name'], 'target': preds})
    submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()
```

## 9. Strategy Priority (IMPORTANT)

1.  **Most Impactful Techniques:** * Utilizing 2018, 2019, and 2020 datasets combined for model training to stabilize CV.
    * Formulating the loss as a 9-class Multiclass Cross-Entropy problem rather than a Binary classification.
    * Ensembling by applying percentile ranking (`pct=True`) prior to blending folds and architectures.
2.  **Secondary Improvements:** * Triple stratified split mechanism preventing patient leakage across folds.
    * Integration of metadata tabular features concatenated onto the pooled CNN embeddings.
3.  **Minor Tricks:** * Massive image data augmentations (Cutout, grid distortions) combined with high-resolution scaling (up to 896x896) relying on PyTorch AMP to manage GPU constraints.