Here is the analysis of the state-of-the-art solution for the Statoil/C-CORE Iceberg Classifier Challenge.

### Data Processing

- **Observation/Action:** The raw flattened radar data (in decibels, dB) is reshaped into 75x75 arrays, normalized using a specific formula `(img + 50) / 60.0`, and clipped to [0, 1]. Missing incidence angles are imputed with 0.0.
- **ML Rationale (Why it works):** Radar backscatter in dB can have extreme negative values. The shift (+50) and scale (/60.0) map the typical operational range of SAR backscatter into a standard [0, 1] range, which is optimal for neural network weight initialization and gradient flow. Clipping prevents extreme outliers (like radar anomalies) from skewing the activation functions.
- **Guiding Principle:** When dealing with non-standard image data (like SAR, medical, or multispectral sensors), apply domain-specific scaling to map the physical units into a normalized [0, 1] or [-1, 1] range before feeding them into standard vision architectures.

### Feature Engineering

- **Observation/Action:** The model creates a 4-channel input. Band 3 is created by subtracting Band 2 from Band 1 (`HH - HV`). Band 4 is created by taking the scalar `inc_angle`, normalizing it, and broadcasting it into a 75x75 spatial channel. 
- **ML Rationale (Why it works):** 
  - *Band 3 (HH - HV):* In SAR imagery, the difference between polarizations highlights the scattering mechanisms of the target. Icebergs and ships reflect radar energy differently across polarizations; their difference acts as a strong discriminative feature.
  - *Band 4 (Incidence Angle):* The background ocean backscatter changes drastically depending on the incidence angle. By broadcasting this scalar into a spatial channel, the CNN/Transformer can naturally condition its spatial filters on the viewing angle without requiring a separate multi-modal dense network.
- **Guiding Principle:** To integrate critical scalar metadata (like sensor angle, time of day, or temperature) into a vision model, broadcast the normalized scalar into a constant spatial channel and increase the input channels of the first convolutional/patch embedding layer.

### Model

- **Observation/Action:** The solution uses a pretrained Swin Transformer (`swin_base_patch4_window7_224`) modified to accept 4 input channels (`in_chans=4`). The 75x75 images are upscaled to 224x224. It utilizes Mixed Precision (AMP), Gradient Accumulation, and Gradient Checkpointing.
- **ML Rationale (Why it works):** Swin Transformers excel at capturing both local textures (crucial for radar speckle and object edges) and global context (ocean background). Upscaling to 224x224 allows the model to leverage ImageNet pretrained weights effectively. Modifying `in_chans=4` directly in `timm` seamlessly adapts the pretrained patch embedding layer to the custom SAR+metadata stack.
- **Guiding Principle:** Do not restrict yourself to 3-channel RGB models. Modern vision libraries allow dynamic input channel adaptation. Upscale low-resolution sensor data to match the expected input size of powerful pretrained architectures to benefit from transfer learning.

### Validation

- **Observation/Action:** The pipeline uses Stratified K-Fold Cross Validation and applies Test-Time Augmentation (TTA) by averaging predictions from the original, horizontally flipped, and vertically flipped images during both validation and inference.
- **ML Rationale (Why it works):** Icebergs and ships viewed from directly above (or via SAR) have no strict "up" or "down" orientation. TTA exploits this rotational/reflectional invariance, smoothing out the model's confidence and reducing log-loss (the competition metric) by providing a more robust, ensemble-like probability estimate.
- **Guiding Principle:** If the target object's orientation is arbitrary relative to the sensor, use geometric Test-Time Augmentation (flips, 90-degree rotations) to stabilize probabilistic predictions and improve metric scores without training multiple models.