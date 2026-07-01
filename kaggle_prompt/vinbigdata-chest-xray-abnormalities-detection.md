## 1. Problem Understanding
- **Task Type:** Object Detection (Medical Imaging). Identify bounding boxes for 14 different abnormalities in chest X-rays.
- **Evaluation Metric:** Mean Average Precision (mAP) at various Intersection over Union (IoU) thresholds.
- **Key Challenges:**
  - High variance and disagreement in ground truth labels (annotated by multiple radiologists).
  - Simulating the "consensus of radiologists" for the hidden test set.
  - Overfitting to the public leaderboard due to inconsistent cross-validation splits across merged teams.
  - Balancing high precision with the need for high recall to maximize the mAP metric (no penalty for adding extra false positive boxes if recall is improved).

## 2. Data Pipeline (Code-Oriented)
- `load_data()`: Read image paths (converted from DICOM to PNG/JPG) and parse the bounding box coordinates, class labels, and radiologist IDs from the CSV.
- `preprocess()`: Resize images to standard dimensions used by the ensemble components (e.g., 512x512 and 640x640). Apply standard normalization (ImageNet means/stds or global dataset means). 
- `feature_engineering()`: Since multiple radiologists annotate the same image, merge overlapping ground truth boxes per image using standard Weighted Boxes Fusion (WBF) before training to create a single clean target per abnormality.
- `split_folds()`: Implement `GroupKFold` or `StratifiedGroupKFold` using the Patient ID as the grouping variable to prevent data leakage across folds. Use 5 folds.

## 3. Model Design
- `build_model(model_name)`: A factory function to instantiate different architectures. To fit in one script, utilize libraries like `timm`, `torchvision`, or `effdet`.
- **Model Types to Support:** - EfficientDet (e.g., D2)
  - ResNet-based Faster R-CNN / Detectron2 equivalents (ResNet50, ResNet101)
  - YOLOv5 (wrapped via PyTorch hub or custom module)
- **Pretrained Usage:** All models must initialize with COCO pre-trained weights.

## 4. Training Strategy
- `train_one_fold()`: Loop through the dataloader. Use Automatic Mixed Precision (AMP) scaler for memory efficiency.
- **Loss Function:** Depending on the base architecture. Focal Loss for classification (to handle class imbalance in rare diseases) and Smooth L1 / GIoU for bounding box regression.
- **Optimizer / Params:** AdamW or SGD with momentum. Use a Cosine Annealing Learning Rate Scheduler with a warmup phase.
- **Tricks:** Apply heavy data augmentation (Mixup, Mosaic, Horizontal Flipped, Random Resized Crop) to prevent overfitting. 

## 5. Validation Strategy
- **Cross-Validation Logic:** Ensure the validation dataloader does NOT use TTA (Test Time Augmentation) during the standard training loop to track pure model performance.
- **OOF Generation:** Save Out-of-Fold (OOF) predictions for each model. This is critical for finding the optimal blending weights for the ensemble later.

## 6. Inference Pipeline
- `predict()`: Run inference on the test set. Implement Test Time Augmentation (TTA) specifically using Horizontal Flip and Multi-Scale testing.
- `ensemble_models()`: Combine predictions from multiple checkpoints (Fully Validated, Partially Validated, Not Validated).
- `post_process()`: Apply the core magic—Modified WBF.

## 7. Key Tricks (ACTIONABLE)
- **Trick 1: Modified WBF (p_sum)** - *Logic:* Standard WBF averages confidences of overlapping boxes. Instead, when IoU > threshold, SUM the confidences (`p_det`). This rewards boxes that multiple models (simulating multiple radiologists) agree upon.
  - *Code Action:* Group overlapping boxes -> Sum their `score` values -> Normalize the final summed scores by dividing by the maximum possible sum or clipping to 1.0.
- **Trick 2: High-Recall Box Injection** - *Logic:* mAP is not penalized for extra boxes.
  - *Code Action:* Take `predictions_A` (from a high mAP model) and `predictions_B` (from a high recall model). Find all boxes in `B` that have an IoU < 0.1 with any box in `A`. Append these non-overlapping boxes from `B` to `A`'s prediction list.
- **Trick 3: Tiered Weighting**
  - *Logic:* Assign different weights in the ensemble based on validation trust.
  - *Code Action:* Assign highest weights to "Fully Validated Stage" models, medium weights to "Partially Validated", and lowest to "Public LB Validated" models to prevent LB overfitting.

## 8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)

```python
import os
import numpy as np
import pandas as pd
import torch
# ... other standard imports (torchvision, albumentations, etc.)

def seed_everything(seed=42):
    """Sets random seeds for reproducibility across numpy, torch, etc."""
    pass

def load_data(csv_path, image_dir):
    """Loads CSV, maps DICOM/image paths, and parses raw bounding boxes."""
    pass

def preprocess_and_augment(is_train):
    """Returns albumentations pipeline for training (Mosaic/Flip) or inference (Resize)."""
    pass

def create_folds(df, n_splits=5):
    """Applies GroupKFold on PatientID to yield fold assignments."""
    pass

def build_model(model_type, num_classes=14):
    """Instantiates EfficientDet, FasterRCNN, or YOLO equivalent based on config."""
    pass

def train_one_fold(fold, model, train_loader, val_loader, device, epochs):
    """Executes the training loop with AMP, loss calculation, and backpropagation."""
    pass

def modified_wbf(boxes_list, scores_list, labels_list, weights, iou_thr=0.4):
    """
    IMPLEMENTS THE MAGIC:
    Blends bounding boxes. For boxes with IoU > iou_thr, sums the confidence 
    scores ('psum') instead of averaging, simulating radiologist consensus. 
    Normalizes final scores.
    """
    pass

def inject_high_recall_boxes(high_map_preds, high_recall_preds, iou_thr=0.1):
    """
    Appends boxes from high_recall_preds to high_map_preds ONLY IF they do not 
    overlap (IoU < iou_thr) with existing boxes.
    """
    pass

def inference(models, test_loader, device, use_tta=True):
    """Runs predictions on test data, applies TTA (horizontal flips), and collects raw boxes."""
    pass

def main():
    # 1. Setup & Configuration
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Data Preparation
    df = load_data('train.csv', 'images/')
    df = create_folds(df)
    
    # 3. Training Pipeline (Representative loop)
    trained_models = []
    for fold in range(5):
        # Initialize loaders
        # model = build_model('efficientdet_d2')
        # model = train_one_fold(fold, model, ...)
        # trained_models.append(model)
        pass # Placeholder for training logic
        
    # 4. Inference & Ensembling Pipeline
    # Assuming trained_models contains different architectures from the 3 validation stages
    test_df = load_data('sample_submission.csv', 'test_images/')
    raw_predictions = inference(trained_models, test_df, device)
    
    # 5. Post-Processing (The Winning Logic)
    final_submissions = []
    for image_preds in raw_predictions:
        # Apply modified WBF (p_sum) to simulate radiologist consensus
        blended_boxes = modified_wbf(
            image_preds['boxes'], 
            image_preds['scores'], 
            image_preds['labels'], 
            weights=[1.0, 0.8, 0.4] # Fully, Partially, Not Validated weights
        )
        final_submissions.append(blended_boxes)
        
    # 6. Save Format
    # save_to_csv(final_submissions, 'submission.csv')

if __name__ == "__main__":
    main()
```

## 9. Strategy Priority (IMPORTANT)

1. **Most impactful techniques:** Implementing the `modified_wbf` with `p_det_weight_psum` to mathematically replicate the hidden test set's radiologist consensus labeling.
2. **Secondary improvements:** Diverse model ensembling (YOLOv5 + Detectron2 + EfficientDet) with explicit weighting based on validation reliability (lowering weights for models only validated on the public LB).
3. **Minor tricks:** Injecting non-overlapping, low-confidence boxes from a high-recall model into the final predictions to game the mAP metric's lack of false-positive penalties.