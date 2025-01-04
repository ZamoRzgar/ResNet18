# ResNet18 for MNIST Classification

## Overview
This project implements a Convolutional Neural Network (CNN) using the ResNet18 architecture for digit recognition on the MNIST dataset. The model leverages residual connections to enhance training efficiency and prevent gradient vanishing issues.

## Features
- Utilizes ResNet18 with residual blocks.
- Supports MNIST dataset for digit classification.
- Implements data preprocessing with resizing, normalization, and grayscale-to-RGB conversion.
- Saves and loads trained model weights for reuse.
- Provides visualization of predictions with matplotlib.

## Dataset
- **MNIST Dataset:**
  - 60,000 training images and 10,000 test images of handwritten digits (0–9).
  - Preprocessed by resizing images to 32x32 and converting grayscale to RGB (3 channels).

## Model Architecture
- **Residual Block:** Includes skip connections for easier gradient flow.
- **ResNet18 Layers:**
  - Initial convolutional layer with ReLU activation.
  - 4 residual layers with increasing channels (64, 128, 256, 512).
  - Average pooling and fully connected output layer.

## Requirements
- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib

Install dependencies:
```bash
pip install torch torchvision numpy matplotlib
```

## Training Parameters
- **Epochs:** 10
- **Batch Size:** 128
- **Learning Rate:** 0.01
- **Momentum:** 0.9
- **Weight Decay:** 5e-4

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/ZamoRzgar/ResNet18.git
   cd ResNet18
   ```
2. Train the model:
   ```bash
   python train.py
   ```
3. Evaluate the model:
   ```bash
   python test.py
   ```
4. Predict on sample images:
   ```bash
   python predict.py
   ```

## Results
- Achieved **99.52% accuracy** on MNIST test dataset.
- Visualized predictions for sample digits.

## Observations
- Residual connections improve learning and prevent gradient vanishing.
- Larger input size (32x32) enhances performance with ResNet.
- Performance may vary with custom datasets or additional augmentations.

## Future Enhancements
- Test model on more diverse datasets.
- Incorporate data augmentation techniques.
- Expand to deeper networks for complex image datasets.

## References
1. He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
2. MNIST dataset: http://yann.lecun.com/exdb/mnist/
3. PyTorch Documentation: https://pytorch.org

---
**Author:** Zamo Rzgar  
**Contact:** Zamo.rzgar1@gmail.com

