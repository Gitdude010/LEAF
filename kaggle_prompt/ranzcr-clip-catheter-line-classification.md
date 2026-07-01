## 1. Problem Understanding
* **Task type:** Multi-label image classification (predicting the presence and correct placement of 11 types of tubes and lines in chest radiographs).
* **Evaluation metric:** Mean Column-wise ROC-AUC.
* **Key challenges:** * Catheters and lines are extremely thin and visually subtle, requiring high-resolution inputs.
    * High-resolution inputs (like 768x768 or 1024x1024) severely constrain GPU memory (VRAM).
    * Not all training images contain pixel-level mask annotations; many only have classification labels.
    * Strict inference time limits require optimized prediction pipelines.

## 2. Data Pipeline (Code-Oriented)
The core philosophy is to leverage segmentation masks as an auxiliary learning signal without requiring them at inference.

* `load_data()`: Read the main CSV and annotation CSV files. Merge them to identify which images have pixel-level coordinate data and which are classification-only.
* `preprocess_masks()`: Iterate through coordinate data. Use OpenCV (`cv2.line`) to draw connected lines between given points on a blank canvas to create target segmentation masks. Set a boolean flag `has_mask` for each record.
* `feature_engineering()`: Implement Multilabel Stratified K-Fold (iterative stratification) based on the 11 target classes and Patient ID to prevent data leakage across folds.
* `get_transforms()`: 
    * **Train:** Albumentations pipeline including `Resize` (to 768x768), `HorizontalFlip`, `ShiftScaleRotate`, `RandomBrightnessContrast`, and `Normalize`. 
    * **Validation/Test:** `Resize` and `Normalize`. 
    * *Note:* Avoid CutMix for standard models; use it strictly for one or two models specifically to inject variance into the final ensemble.

## 3. Model Design
The architecture must force the convolutional backbone to physically locate the lines using segmentation, while ultimately solving a classification problem.

* `build_model()`: Construct a Dual-Head architecture comprising an encoder, a lightweight decoder, and a classification head.
* **Encoder (Backbone):** Utilize `timm` library for `resnet200d`, `tf_efficientnet_b5_ns`, and `tf_efficientnet_b7_ns`.
* **Decoder (Segmentation Head):** A minimal UNet decoder attached to the encoder's feature maps. Compress the channel dimensions of the decoder blocks aggressively to save VRAM. Outputs a 1-channel mask.
* **Classification Head:** Global Average Pooling on the encoder's final feature map, followed by a linear layer outputting 11 logits.
* **Forward Pass Logic:** * `if self.training:` Return `(classification_logits, segmentation_mask)`.
    * `if not self.training:` Return `classification_logits` ONLY (bypassing the decoder entirely to save memory and compute).

## 4. Training Strategy
Training is executed in staged resolutions to bootstrap high-quality pseudo-masks.

* `train_one_fold()`: Standard PyTorch training loop utilizing mixed precision (`torch.cuda.amp`) to handle 768+ resolutions.
* **Loss Function:** A composite loss $L_{total} = L_{BCE\_cls} + \lambda \cdot L_{BCE\_seg}$. 
    * *Crucial Logic:* During the forward pass, check the `has_mask` flag. If the image lacks annotation data, set $\lambda = 0$ for that specific item, applying only the classification loss.
* **Optimizer:** `AdamW` with a `CosineAnnealingLR` scheduler. Use gradient clipping to prevent explosion during early epochs when both heads are stabilizing.
* **Staged Execution Plan:**
    1.  Train at 512x512 using only provided annotations.
    2.  Run inference on the *entire* training set to generate "v1 pseudo-masks" for unannotated images.
    3.  Scale up to 768x768 and retrain using the v1 pseudo-masks for all images.
    4.  (Optional) Scale up to 1024x1024 for a final ResNet200D run.

## 5. Validation Strategy
* **Cross-Validation:** Standard 5-fold CV.
* **Metrics Calculation:** Generate Out-Of-Fold (OOF) predictions exclusively from the classification head. Calculate Mean Column-wise ROC-AUC using `sklearn.metrics.roc_auc_score`.
* **Checkpointing:** Ignore segmentation metrics for checkpointing; save model weights strictly based on the highest validation ROC-AUC score.

## 6. Inference Pipeline
* `predict()`: Load the saved dual-head weights. 
* **Memory Optimization:** Ensure the UNet decoder is fundamentally bypassed or physically deleted from the model graph before inference begins to meet the 9-hour Kaggle runtime limit.
* **TTA (Test Time Augmentation):** Apply Horizontal Flip (HFlip) to the batch, run inference, reverse the flip on the predictions, and average with the original predictions.
* **Ensemble:** Concatenate the probability outputs of the `resnet200d`, `efficientnet-b5`, and `efficientnet-b7` models and take the unweighted mean.

## 7. Key Tricks (ACTIONABLE)
* **If VRAM is exhausted ->** drastically reduce the feature channels in the UNet decoder. The segmentation output does not need to be perfect; it merely acts as an attention mechanism for the backbone.
* **If an image has no mask ->** zero out the segmentation loss for that batch item dynamically: `seg_loss = seg_loss * mask_available_tensor`.
* **If inference is timing out ->** enforce `model.decoder = nn.Identity()` or explicitly write the forward pass to ignore the decoder prior to the inference loop.
* **Advanced Retrieval Post-Processing (Optional LB Boost):** Train a metric learning model on patient IDs using an external dataset, extract embeddings, and use KNN to find the most similar external image to adjust border-line predictions. (Keep this out of the core pipeline unless baseline metrics are maxed out).


## 8. Strategy Priority (IMPORTANT)
1. Most Impactful Techniques (Core Architecture & Pipeline)

Dual-Head Training with Auxiliary Loss: Using segmentation masks as a supervisor forces the model's attention onto the extremely subtle lines/catheters, significantly boosting classification accuracy.

Staged Pseudo-Labeling for Masks: Bootstrapping the dataset by training at 512x512 on annotated data, generating full-dataset pseudo-masks, and then retraining at 768x768 using the complete mask dataset.

2. Secondary Improvements (Scaling & Robustness)

High-Resolution & Heavy Backbones: Scaling up to 768x768 (and up to 1024x1024) using robust architectures like resnet200d and efficientnet-b7.

Ensembling Diverse Models: Averaging predictions from different model families (ResNet vs. EfficientNet) and different resolutions to reduce variance.

Test-Time Augmentation (TTA): Applying Horizontal Flip (HFlip) during inference to stabilize and marginally improve predictions.

3. Minor Tricks (Optimization & Edge Cases)

Dynamic Loss Masking: Dynamically multiplying the segmentation loss by 0 during the forward pass for images that lack mask annotations, preventing noisy gradients.

Inference VRAM/Time Optimization: Using a severely bottlenecked UNet decoder to save VRAM during training, and completely bypassing/dropping the decoder during inference to strictly meet the 9-hour Kaggle limit.

KNN Image Retrieval (Post-Processing): Using an external metric learning model to retrieve similar images from the ChestX dataset to adjust final borderline predictions (high engineering effort for a final LB push).