# Enrollment-Conditioned Conv-TasNet for Real-Time Speech Extraction

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📌 Overview
This repository implements a **Speaker-Conditioned Conv-TasNet** designed for targeted speech extraction. Unlike traditional "blind" separation, this system utilizes a speaker enrollment mechanism (**ECAPA-TDNN**) to isolate a specific target speaker from a monaural mixture in real-time.

This project was developed as a Final Year Engineering Project at **NUST**, focusing on low-latency audio processing and speaker-dependent mask estimation.

## 🚀 Key Features
* **Targeted Separation:** Uses 192-dimensional speaker embeddings to guide the extraction process.
* **FiLM Conditioning:** Implements Feature-wise Linear Modulation (FiLM) to steer the TCN blocks toward the target identity.
* **Real-Time Ready:** Includes optimized inference scripts for low-latency hardware like laptops and Raspberry Pi.
* **ONNX Integration:** Model exported for high-performance cross-platform execution.

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `conditioned_convtasnet.py` | Core architecture: Conv-TasNet fused with conditioning layers. |
| `Dataset_class.py` | Custom PyTorch data loader with dynamic mixing and augmentation. |
| `train.py` | Training pipeline with differential learning rates and Cosine Annealing. |
| `continuous_test.py` | Real-time audio extraction using `sounddevice` and GPU acceleration. |
| `analysis_1.py` | Quantitative analysis script for calculating SI-SNR gain on recorded samples. |
| `model.onnx` | Exported model for edge deployment. |

## 📊 Datasets
The model is trained on a massive corpus combining diverse speaker profiles:
* **LibriSpeech (Clean-100):** [OpenSLR-12](https://openslr.trmal.net/resources/12/train-clean-100.tar.gz) - Baseline English speech.
* **Custom Fine-Tuning Corpus:** [Google Drive Download](https://drive.google.com/file/d/1y8SxqKo0EAO3UJM3bzTpz8bRFVuGougO/view?usp=sharing) - Specialized data for targeted extraction.

## ⚙️ Installation & Usage

### 1. Setup Environment
```bash
git clone [https://github.com/Haris-Khan14/enrollment-conv-tasnet.git](https://github.com/Haris-Khan14/enrollment-conv-tasnet.git)
cd enrollment-conv-tasnet
pip install torch torchaudio speechbrain asteroid sounddevice
