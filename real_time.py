import torch
import torchaudio
import sounddevice as sd
import numpy as np
import os

from conditioned_convtasnet import ConditionedConvTasNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_RATE = 8000

# -----------------------
# Load Model
# -----------------------

model = ConditionedConvTasNet()

checkpoint = torch.load("best_model.pth", map_location=device)

model.convtasnet.load_state_dict(checkpoint["convtasnet"])
model.film_layers.load_state_dict(checkpoint["film_layers"])

model.convtasnet.to(device)
model.film_layers.to(device)

model.eval()

# -----------------------
# Record Enrollment
# -----------------------

print("\nSpeak for enrollment (3 seconds)... CLEARLY AND CLOSE TO MIC")
enroll_audio = sd.rec(int(3 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
sd.wait()

# NEW: Force Normalize Enrollment before embedding extraction
enrollment = torch.tensor(enroll_audio.T).float()
enrollment = enrollment - enrollment.mean()
enrollment = enrollment / (enrollment.abs().max() + 1e-8) # Make it loud and clear

print("Enrollment recorded.\n")

# -----------------------
# Record Mixture
# -----------------------

print("Now play TWO speakers for 10 seconds (you + Gul)...")

mixture_audio = sd.rec(
    int(10 * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1
)

sd.wait()

print("Mixture recorded.\n")

# -----------------------
# Convert to Torch
# -----------------------
enrollment = torch.tensor(enroll_audio.T).float()
mixture = torch.tensor(mixture_audio.T).float()

# ONLY remove the mean to center the waveform
mixture = mixture - mixture.mean()
enrollment = enrollment - enrollment.mean()

mixture_tensor = mixture.unsqueeze(0).to(device)
enroll_tensor = enrollment.unsqueeze(0).to(device)

# -----------------------
# Run Model
# -----------------------
with torch.no_grad():
    output = model(mixture_tensor, enroll_tensor)

output = output.squeeze(0).cpu()

# -----------------------
# High-Quality Polish
# -----------------------
# 1. Peak Normalization
max_val = output.abs().max()
if max_val > 1e-4:
    output = output / max_val

# 2. Low-Pass Filter (Removes high-frequency digital "crackle")
# This keeps only frequencies below 3800Hz, which is where speech lives.
output = torchaudio.functional.lowpass_biquad(output, SAMPLE_RATE, cutoff_freq=3800)

# 3. Final safety clip
output = torch.clamp(output, -0.9, 0.9)

# -----------------------
# Save Files
# -----------------------
os.makedirs("realtime_outputs", exist_ok=True)

# Important: save as float32 to prevent bit-depth clipping
torchaudio.save("realtime_outputs/mixture.wav", mixture, SAMPLE_RATE)
torchaudio.save("realtime_outputs/extracted.wav", output, SAMPLE_RATE)
torchaudio.save("realtime_outputs/enrollment.wav", enrollment, SAMPLE_RATE)

print("\nFiles saved to realtime_outputs/extracted.wav")