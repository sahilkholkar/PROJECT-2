# Anomaly Detection in Surveillance Videos using Convolutional Autoencoder

## 📹 Overview
This project uses a deep learning-based autoencoder to detect anomalies (e.g., falls, sneak, hit) in CCTV surveillance footage. The model is trained only on normal activity to learn reconstruction patterns, and deviations are flagged as anomalies.

## 💡 Features
- Frame extraction and preprocessing using OpenCV
- Convolutional Autoencoder trained on normal patterns
- SSIM and MSE loss for accurate reconstruction error analysis
- 95th percentile thresholding for anomaly detection
- Real-time compatible design with <500ms inference latency

## 🧱 Architecture
- **Input**: Grayscale surveillance video frames
- **Model**: PyTorch-based autoencoder with 4-layer encoder-decoder
- **Detection**: Anomaly score based on SSIM loss + threshold
- **Visualization**: Reconstructed frame comparison and anomaly heatmaps

## 📊 Results
- ROC AUC: 0.91 | Precision: 85%
- Detected rare actions with minimal false positives
- Real-time detection pipeline validated on GPU

## 🔧 Setup
```bash
pip install torch torchvision opencv-python matplotlib numpy
