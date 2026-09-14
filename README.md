# 🛰️ Aerial Image Segmentation with U-Net

Semantic segmentation model that classifies aerial imagery into 6 land-use categories using a U-Net convolutional neural network architecture.

## What it does

1. **Loads and preprocesses** a dataset of aerial images with their corresponding segmentation masks.
2. **Splits images into patches** (160x160 px) to make training feasible on standard hardware.
3. **Maps RGB mask colors to class labels**: Unlabelled, Land, Road, Vegetation, Water, Building.
4. **Trains a full U-Net architecture** (encoder-decoder with skip connections) built from scratch in TensorFlow/Keras.
5. **Evaluates performance** using accuracy and Jaccard Index (IoU) as custom metrics.

## Results

| Metric | Training | Validation |
|---|---|---|
| Accuracy | 82.1% | **83.8%** |
| Jaccard Index (IoU) | 65.3% | **66.4%** |

Trained for 5 epochs on a 90/10 train/validation split. Both metrics improved consistently across epochs with no signs of overfitting (validation accuracy stayed on par with or above training accuracy).

## Tech Stack

- **TensorFlow / Keras** — model architecture and training
- **OpenCV** — image loading and patch extraction
- **NumPy** — array manipulation and label encoding
- **scikit-learn** — train/test split
- **Matplotlib** — visualization of predictions

## Files

- `Si.py` — full training pipeline: data loading, U-Net architecture, training loop, and evaluation
- `training_history.csv` — epoch-by-epoch training metrics (accuracy, loss, Jaccard Index)

## Notes

The trained model checkpoint (~363 MB) is not included in this repository due to size constraints, but is available upon request.
