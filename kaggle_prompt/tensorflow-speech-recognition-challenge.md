## 1. Problem Understanding
* **Task type:** Multiclass Audio Classification (Speech Command Recognition). Identifying specific spoken words, plus handling an "unknown" class and a "silence" background class.
* **Evaluation metric:** Categorical Accuracy.
* **Key challenges:** * Severe data domain shift between training/validation sets and the test (Leaderboard) set.
    * Distribution mismatches for the "unknown" and "silence" categories.
    * Sensitivity of neural networks to class imbalances during training.
    * Strict size and inference speed constraints (if targeting the special prize track).

## 2. Data Pipeline (Code-Oriented)
* **`load_data(data_dir)`:** Traverse the directory structure, parse `.wav` files using `librosa` or `scipy.io.wavfile` at a fixed sample rate (e.g., 16kHz). Map folder names to integer class labels.
* **`preprocess(audio_array)`:** Standardize sequence lengths. Pad audio arrays shorter than 1 second with zeros; randomly crop or truncate arrays longer than 1 second.
* **`feature_engineering(audio_array, feature_type)`:** Implement a switch to generate one of three input types based on configuration:
    1.  `raw_1d`: Return the normalized 1D waveform.
    2.  `log_mel`: Extract log-melspectrograms using `librosa.feature.melspectrogram` and logarithmic scaling.
    3.  `mfcc`: Extract Mel-frequency cepstral coefficients.
* **`split_folds(df)`:** Implement a robust validation split. Do not use random splits. Group by speaker ID (extracted from filenames) to ensure the same speaker does not appear in both train and validation sets, simulating the unseen test environment.

## 3. Model Design
* **`build_model(input_shape, model_type)`:** Use TensorFlow/Keras to construct the network graph.
* **Model types:**
    * *Type A (Raw Waveform):* 1D Convolutional Neural Network utilizing `DepthwiseConv1D` layers. This drastically reduces the parameter count to stay under the 1.25M parameter limit while capturing temporal audio patterns.
    * *Type B (Spectrogram/MFCC):* 2D Convolutional Neural Networks treating the audio features as single-channel images (e.g., simplified VGG or ResNet architectures).
* **Pretrained usage:** Train from scratch. Model Distillation (Teacher-Student) is applied later, where a massive ensemble (Teacher) generates soft labels to train a lightweight 1D CNN (Student).

## 4. Training Strategy
* **`train_one_fold()`:** Wrap the `input_data.py` logic or build a custom `tf.keras.utils.Sequence` generator.
* **Loss function:** Categorical Crossentropy. If using distillation, use Kullback-Leibler Divergence for soft targets.
* **Optimizer / params:** Stochastic Gradient Descent (SGD) with Nesterov Momentum.
* **Tricks:** * *Balanced Class Sampling:* The data generator MUST force a balanced distribution of known, unknown, and silence classes in every batch.
    * *Pseudo-Labeling Loop:* Identify test samples with >0.6 prediction probability from a baseline model. Add these to the training generator. Schedule the generator to yield 100% pseudo-labels for epochs 1-5, then mix with true training data for the remaining epochs.

## 5. Validation Strategy
* **Cross-validation logic:** Evaluate models on both the standard validation fold and a tracked subset of the Leaderboard data (using highly confident pseudo-labels as a proxy validation set).
* **OOF generation:** Save Out-Of-Fold probability matrices to calculate local validation accuracy and for subsequent ensembling/stacking. A model is only accepted if it improves both the local validation metric and the pseudo-leaderboard metric.

## 6. Inference Pipeline
* **`predict(model, test_generator)`:** Iterate through the test `.wav` files and output softmax probability arrays.
* **TTA (Test Time Augmentation):**
    * Generate 3-5 variants of each test sample: Time-shifted (roll array), Volume adjusted (multiply amplitude), and Time-stretched (`librosa.effects.time_stretch`).
    * Run inference on all variants and average the resulting softmax arrays.
* **Ensemble:** Combine the 1D waveform, Log-Mel, and MFCC model predictions using a weighted average. Derive weights from the score distribution (confidence margin).

## 7. Key Tricks (ACTIONABLE)
* **If Validation != LB:** Implement aggressive pseudo-labeling. Filter test predictions using class-specific thresholds (e.g., 0.6 for knowns, 0.8 for silence) to ensure the newly injected training data remains balanced.
* **If targeting fast inference/low size:** Swap standard `Conv1D` for `SeparableConv1D` or `DepthwiseConv1D`.
* **If mixing frameworks:** Use TF backend utilities to wrap the native Keras models, allowing easy graph extraction and freezing for deployment.
* **Hyperparameters:** Use a basic SGD optimizer but ensure momentum is enabled (e.g., `momentum=0.9`).

## 8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)

```python
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import librosa

def seed_everything(seed=42):
    # Set random seeds for numpy, tensorflow, python.random to ensure reproducibility.
    pass

def load_data(data_dir):
    # Parse directory, extract file paths and labels, handle speaker IDs.
    pass

def preprocess(audio_path, target_length=16000):
    # Load audio, pad with zeros or truncate to exact target_length.
    pass

def feature_engineering(audio_array, feat_type='raw'):
    # Branching logic: return raw 1d, compute log_mel via librosa, or compute mfcc.
    pass

class BalancedAudioGenerator(tf.keras.utils.Sequence):
    # Custom Keras generator to ensure equal class representation per batch.
    # Handles dynamic application of pseudo-labels and training mix schedules.
    pass

def create_folds(df, num_folds=5):
    # Split dataframe using GroupKFold based on the speaker_id column.
    pass

def build_model(input_shape, num_classes, model_type='1d_depthwise'):
    # Construct and compile Keras model (1D CNN with Depthwise layers or 2D CNN).
    pass

def train_one_fold(fold_idx, train_df, val_df, pseudo_df=None):
    # Initialize BalancedAudioGenerator.
    # Build model, define callbacks (ModelCheckpoint, ReduceLROnPlateau).
    # Fit model using fit_generator. Return trained model and OOF predictions.
    pass

def apply_tta(audio_array):
    # Apply time shift, volume scale, and time stretch. Return list of augmented arrays.
    pass

def inference(models_list, test_df):
    # Iterate test set, apply TTA, run model.predict(), average TTA results.
    # Average across all models in models_list. Return final probabilities.
    pass

def generate_pseudo_labels(predictions, test_df, threshold=0.6):
    # Filter test_df based on prediction confidence > threshold. Return df for next training loop.
    pass

def main():
    seed_everything(42)
    train_data, test_data = load_data('./dataset')
    
    # Feature Extraction
    # ... applies feature_engineering to dataframes ...
    
    folds = create_folds(train_data)
    
    # First Pass: Train baseline models
    trained_models = []
    for fold in range(len(folds)):
        model = train_one_fold(fold, train_data[train], train_data[val])
        trained_models.append(model)
        
    # Generate initial predictions on Test set
    test_preds = inference(trained_models, test_data)
    
    # Pseudo Labeling Loop
    pseudo_data = generate_pseudo_labels(test_preds, test_data, threshold=0.6)
    
    # Second Pass: Train with pseudo-labels
    final_models = []
    for fold in range(len(folds)):
        model = train_one_fold(fold, train_data[train], train_data[val], pseudo_df=pseudo_data)
        final_models.append(model)
        
    # Final Inference and TTA
    final_preds = inference(final_models, test_data)
    
    # Save submission
    # ... formatting to CSV ...

if __name__ == "__main__":
    main()
```

## 9. Strategy Priority (IMPORTANT)

1.  **Most impactful techniques:** Balanced class sampling in the data generator & iterative Pseudo-labeling carefully filtered by class-specific probability thresholds.
2.  **Secondary improvements:** Ensembling disparate model architectures (1D raw, 2D Log-Mel, 2D MFCC) & applying Test Time Augmentation (Time stretch, pitch shift, volume adjust).
3.  **Minor tricks:** Utilizing Depthwise1D convolutions for footprint reduction & replacing Adam with standard SGD + Momentum.