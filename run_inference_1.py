import torch
import torchaudio
import os
from conditioned_convtasnet import ConditionedConvTasNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Model
model = ConditionedConvTasNet()
checkpoint = torch.load("best_model.pth", map_location=device)
model.convtasnet.load_state_dict(checkpoint["convtasnet"])
model.film_layers.load_state_dict(checkpoint["film_layers"])

model.convtasnet.to(device)
model.film_layers.to(device)
model.eval()

# 2. Helper Audio Loader
def load_audio(path, target_sr=8000):
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    # Ensure Mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav

# 3. Load Files
gul = load_audio("model_test_1/Gul_00059.wav")
haris_target = load_audio("model_test_1/Haris_00045.wav")
haris_enroll = load_audio("model_test_1/Haris_00055.wav")

# Match lengths
min_len = min(gul.shape[-1], haris_target.shape[-1])
gul = gul[:, :min_len]
haris_target = haris_target[:, :min_len]

# Create mixture
snr_db = 0
alpha = 10 ** (-snr_db / 20)
mixture = gul + alpha * haris_target

# Add batch dimension and move to device
mixture_tensor = mixture.unsqueeze(0).to(device)
enrollment_tensor = haris_enroll.unsqueeze(0).to(device)

# 4. Inference
with torch.no_grad():
    # Pass tensors to model
    output = model(mixture_tensor, enrollment_tensor)

# 5. Post-Processing & Normalization
output = output.squeeze(0).cpu()

# Normalize volume to avoid clipping (crucial for "robotic" sound)
output = output / (output.abs().max() + 1e-8)

# 6. Save Files
os.makedirs("outputs", exist_ok=True)
torchaudio.save("outputs/mixture.wav", mixture, 8000)
torchaudio.save("outputs/extracted.wav", output, 8000)

print("Saved files to outputs/ - Check if distortion is reduced.")