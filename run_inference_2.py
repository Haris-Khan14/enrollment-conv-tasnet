import torch
import torchaudio
import os
from conditioned_convtasnet import ConditionedConvTasNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# 1. Load Model
# -------------------------
model = ConditionedConvTasNet()
checkpoint = torch.load("best_model.pth", map_location=device)
model.convtasnet.load_state_dict(checkpoint["convtasnet"])
model.film_layers.load_state_dict(checkpoint["film_layers"])

model.convtasnet.to(device)
model.film_layers.to(device)
model.eval()

# -------------------------
# 2. Helper Loader
# -------------------------
def load_audio(path, target_sr=8000):
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav

# -------------------------
# 3. Load Files (model_test_2)
# -------------------------
haris = load_audio("model_test_2/Haris_00045.wav")
gul_target = load_audio("model_test_2/Gul_00059.wav")
gul_enroll = load_audio("model_test_2/Gul_00095.wav")

# Match lengths
min_len = min(haris.shape[-1], gul_target.shape[-1])
haris = haris[:, :min_len]
gul_target = gul_target[:, :min_len]

# Create mixture (equal SNR)
snr_db = 0
alpha = 10 ** (-snr_db / 20)
mixture = haris + alpha * gul_target

# Batch dimension
mixture_tensor = mixture.unsqueeze(0).to(device)
enrollment_tensor = gul_enroll.unsqueeze(0).to(device)

# -------------------------
# 4. Inference
# -------------------------
with torch.no_grad():
    output = model(mixture_tensor, enrollment_tensor)

output = output.squeeze(0).cpu()

# Normalize
output = output / (output.abs().max() + 1e-8)

# -------------------------
# 5. Save
# -------------------------
os.makedirs("outputs_2", exist_ok=True)

torchaudio.save("outputs_2/mixture.wav", mixture, 8000)
torchaudio.save("outputs_2/extracted_gul.wav", output, 8000)
torchaudio.save("outputs_2/clean_gul.wav", gul_target, 8000)

print("Saved files to outputs_2/")