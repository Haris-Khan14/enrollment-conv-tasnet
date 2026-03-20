# Enrollment-Conditioned Conv-TasNet for Real-Time Speech Extraction

## Overview
This repository implements a Speaker-Conditioned Conv-TasNet (Convolutional Time-domain Audio Separation Network) designed for targeted speech extraction. Unlike traditional "blind" separation, this system uses a speaker enrollment mechanism (ECAPA-TDNN) to isolate a specific target speaker from a monaural mixture in real-time.

This project is part of a final year engineering effort focused on low-latency, high-fidelity audio processing and speaker-dependent mask estimation.

## Key Features
 Targeted Separation: Uses speaker embeddings to condition the mask generation, ensuring the model isolates the desired speaker rather than a random source.
 Real-Time Inference: Optimized scripts for live audio stream processing.
 ECAPA-TDNN Integration: Leverages state-of-the-art speaker encoders for robust identity embeddings.
 Loss Optimization: Implementations of SI-SNR (Scale-Invariant Source-to-Noise Ratio) and targeted separation losses.
 Modular Design: Separate classes for dataset handling, loss calculation, and model architecture to facilitate easy experimentation.

## Repository Structure

| File/Folder | Description |
| :--- | :--- |
| `pretrained_ecapa/` | Contains the pre-trained weights for the ECAPA speaker encoder. |
| `conditioned_convtasnet.py` | Core architecture: The Conv-TasNet model with conditioning layers. |
| `Dataset_class.py` | Custom PyTorch data loader for processing mixtures and speaker references. |
| `real_time.py` | Implementation for low-latency, real-time audio extraction. |
| `train.py` | Main training pipeline for the conditioned model. |
| `losses.py` | Implementations of SI-SNR and other specialized objective functions. |
| `inspect_masker.py` | Utility to visualize and debug the learned masks. |
| `test_.py` | Suite of test scripts for unit testing the dataset, ECAPA encoder, and model. |

## Getting Started

### Prerequisites
 Python 3.8+
 PyTorch (CUDA recommended)
 Torchaudio
 Librosa
 SoundFile

### Installation
```bash
git clone https://github.com/Haris-Khan14/enrollment-conv-tasnet.git
cd enrollment-conv-tasnet
pip install -r requirements.txt
```

### Usage

1. Training the Model:
Configure your dataset paths in `train.py` and run:
```bash
python train.py
```

2. Running Inference:
To extract a speaker from a file using a reference enrollment:
```bash
python run_inference_1.py --input mixture.wav --reference enrollment.wav
```

3. Real-Time Testing:
To test the model's performance on a simulated real-time stream:
```bash
python real_time.py
```

## Technical Details
The system follows a three-stage process:
1.  Encoder: High-dimensional representation of the mixture waveform.
2.  Conditioning: The target speaker's ECAPA-TDNN embedding is fused with the bottleneck features of the separator.
3.  Masker: Generation of a source-specific mask applied to the encoder output.
4.  Decoder: Reconstructs the target speaker's time-domain waveform.

## Author
Haris Khan | Hassan Nasir | Shumaila Asif
Final Year BEE Student | NUST (National University of Sciences and Technology), Pakistan
