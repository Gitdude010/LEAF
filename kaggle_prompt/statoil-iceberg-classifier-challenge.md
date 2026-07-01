## 1. Problem Understanding
* **Task Type:** Binary Image Classification (Iceberg vs. Ship) using satellite imagery and metadata.
* **Evaluation Metric:** Logarithmic Loss (Log Loss). Highly sensitive metric where overconfident, incorrect predictions incur massive penalties.
* **Key Challenges:** Small training dataset, high noise in satellite imagery, extreme disparity between local CV and public Leaderboard (only 20% of test data), and avoiding extreme probabilistic outputs to prevent catastrophic log loss penalties.

## 2. Data Pipeline (Code-Oriented)
The core logic relies on exploiting the metadata (`inc_angle`) to create data subgroups.

* `load_data()`: Read image bands and metadata. Convert string representations of `inc_angle` to float. Handle missing values (e.g., replace 'na' with interpolation or an extreme outlier value, though clustering makes median imputation safer).
* `preprocess_images()`: Combine image bands into standard 3-channel tensors (e.g., Band 1, Band 2, Band 1/Band 2 average). Apply standard normalization.
* `feature_engineering_inc_angle()`: 
    * Round `inc_angle` to 4 decimal places.
    * Map identical 4-decimal `inc_angle` values to common IDs.
    * Run a clustering algorithm (e.g., `sklearn.cluster.AgglomerativeClustering` with Ward linkage or `DBSCAN`) on the 1D `inc_angle` array.
    * Identify the two distinct distributions mentioned by the winners. Label the alternating, pure-iceberg pattern as `is_group_1=True`, and the dense-center distribution as `is_group_1=False` (Group 2).
* `split_folds()`: Implement `StratifiedKFold` (5 or 10 folds). Ensure that identical `inc_angle` samples are kept strictly within the same fold to prevent target leakage during validation.

## 3. Model Design
To mimic the "100+ models" without timing out a single script, implement a loop over a dictionary of 3-5 diverse architectures.

* `build_model(backbone_name, include_angle)`:
    * Initialize a pretrained CNN backbone (e.g., `resnet34`, `densenet121`, `vgg16`).
    * Extract image features using the CNN head.
    * If `include_angle=True`: Pass the `inc_angle` through a small dense layer, then concatenate it with the flattened CNN image features.
    * Pass the concatenated vector through fully connected layers (e.g., Linear -> ReLU -> Dropout -> Linear) to output a single logit.
* **Model Types:** Use a mix of heavy and light architectures (e.g., `tf_efficientnet_b0`, `resnet34`, `densenet121`).

## 4. Training Strategy
The winning strategy isolates subsets of data for targeted training.

* `train_one_fold()`: 
    * **Phase 1 (Optional but recommended for stacking):** Train base models on all data to establish baseline representations.
    * **Phase 2 (The Winning Differentiator):** Filter the training data to `is_group_1 == False` (Group 2). Train a separate suite of CNNs *exclusively* on this subset to maximize signal-to-noise ratio for the difficult samples.
* **Loss Function:** `BCEWithLogitsLoss`.
* **Optimizer:** AdamW with a cosine annealing learning rate scheduler.
* **Tricks:** * Apply Automatic Mixed Precision (AMP) to speed up training.
    * Apply label smoothing (e.g., 0.05) natively during training to prevent the network from pushing logits to extreme values, naturally hedging against the log loss metric.

## 5. Validation Strategy
* **Cross-Validation Logic:** Public LB is untrustworthy (only 20% of test). Rely entirely on Out-Of-Fold (OOF) log loss.
* **OOF Generation:** Store OOF predictions for every model. Stack these OOF predictions to find optimal blending weights using Ridge Regression or a simple bounded Nelder-Mead optimization. Calculate metrics independently for Group 1 and Group 2.

## 6. Inference Pipeline
* `predict_group2()`: Run the models trained exclusively on Group 2 data to predict the test samples falling into Group 2. Apply Test Time Augmentation (TTA) via horizontal/vertical flips.
* `predict_group1()`: For test samples identified as Group 1 via `inc_angle` clustering, bypass the CNNs entirely. Hardcode their predictions to a high-confidence, but safe, iceberg value.
* `post_process_predictions()`: 
    * **Leakage Transfer:** Identify test samples that share an exact 4-decimal `inc_angle` with a training sample. If their CNN prediction is within a specific threshold delta, pull the prediction closer to the known training label.
    * **Clipping:** Run `np.clip(predictions, 0.01, 0.99)` to ensure no prediction is exactly 0.0 or 1.0, avoiding infinite log loss penalties.

## 7. Key Tricks (ACTIONABLE)
* **Trick 1:** `if test_inc_angle in train_inc_angles_rounded_to_4:` -> Blend prediction heavily towards the mode of the training label for that specific angle.
* **Trick 2:** `if sample_in_group_1:` -> `pred = 0.99` (Override CNN prediction completely, as Group 1 is empirically 100% icebergs, but do not use 1.0).
* **Trick 3:** `clip_bounds = (0.005, 0.995)` -> Apply universally to final submission array.

## 8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)

```python
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import AgglomerativeClustering
import torch
import torch.nn as nn
# ... other standard imports ...

def seed_everything(seed=42):
    # Fix all RNG seeds for reproducibility

def load_data():
    # Load train.json and test.json
    # Convert 'na' inc_angle to NaN, fill with median
    # Return raw dataframes

def extract_and_format_images(df):
    # Reshape lists into 75x75x2 arrays
    # Create 3rd channel (e.g., mean of band 1 and 2)
    # Return normalized numpy arrays

def cluster_inc_angles(train_df, test_df):
    # Concat train and test inc_angles
    # Round to 4 decimal places
    # Run AgglomerativeClustering (or custom density logic)
    # Assign 'is_group_1' boolean flag to both dataframes
    # Return updated dataframes

def create_folds(df, n_splits=5):
    # GroupKFold or StratifiedKFold based on inc_angle string representation
    # Ensure identical angles don't cross train/val boundaries

def build_model(arch_name, use_angle_feature=True):
    # Initialize timm model / custom CNN
    # Replace classifier head to accept concatenated (CNN features + dense angle feature)
    # Return PyTorch model

def train_one_fold(fold, train_loader, val_loader, model_config):
    # Setup BCE loss, AdamW, Cosine Scheduler
    # Standard PyTorch training loop with AMP
    # Save best model based on validation log_loss
    # Return best model and OOF predictions

def train_group2_models(df, folds):
    # Filter df where is_group_1 == False
    # Loop through diverse model configurations (e.g., resnet, densenet)
    # Call train_one_fold() for each
    # Return ensemble of trained models and their OOFs

def predict_test_group2(models, test_loader):
    # Run inference with TTA (flips)
    # Average predictions across ensemble
    # Return raw probabilities

def apply_leakage_and_heuristics(test_df, train_df, base_preds):
    # 1. For samples where is_group_1 == True, override base_preds to 0.99
    # 2. Match 4-decimal inc_angles between train and test
    # 3. Pull predictions toward known train labels if deviation threshold is met
    # 4. Return heavily modified predictions

def clip_predictions(preds, lower=0.01, upper=0.99):
    # np.clip(preds, lower, upper)
    # Return final safe predictions

def main():
    seed_everything()
    
    # Data Pipeline
    train_df, test_df = load_data()
    train_images = extract_and_format_images(train_df)
    test_images = extract_and_format_images(test_df)
    train_df, test_df = cluster_inc_angles(train_df, test_df)
    folds = create_folds(train_df)
    
    # Model Training (Group 2 specific)
    group2_models = train_group2_models(train_df, folds)
    
    # Inference
    test_group2_loader = create_dataloaders(test_df)
    raw_preds = predict_test_group2(group2_models, test_group2_loader)
    
    # Post-processing & Magic
    heuristic_preds = apply_leakage_and_heuristics(test_df, train_df, raw_preds)
    final_preds = clip_predictions(heuristic_preds)
    
    # Output
    save_submission(test_df['id'], final_preds)

if __name__ == "__main__":
    main()
```
*Function Explanations:*
* `cluster_inc_angles()`: Implements the crucial 1st place discovery, mathematically splitting the dataset into the pure iceberg group and the mixed group based solely on telemetry data.
* `train_group2_models()`: Executes the targeted training step that maximizes signal-to-noise by ignoring the easily separated Group 1 data during CNN optimization.
* `apply_leakage_and_heuristics()`: Fuses the output of the CNNs with the deterministic rules discovered during exploratory data analysis, overriding neural network uncertainty with hard data.

## 9. Strategy Priority (IMPORTANT)

1.  **Most impactful techniques:** Splitting the dataset via `inc_angle` clustering and retraining the CNN ensemble *exclusively* on Group 2. Hard-coding Group 1 test predictions based on training distribution.
2.  **Secondary improvements:** Exploiting the 4-decimal exact match leakage between training and test sets to post-process predictions.
3.  **Minor tricks:** Clipping final predictions to avoid log_loss decimation, and using diverse CNN architectures (mixing models that use `inc_angle` in the fully connected layer with models that don't) for the ensembling phase.