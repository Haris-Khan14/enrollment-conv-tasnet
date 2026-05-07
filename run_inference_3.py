import torch
import torchaudio
import os
from conditioned_convtasnet import ConditionedConvTasNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ConditionedConvTasNet()
checkpoint = torch.load("best_model_finetuned.pth", map_location=device)
model.convtasnet.load_state_dict(checkpoint["convtasnet"])
model.film_layers.load_state_dict(checkpoint["film_layers"])

model.convtasnet.to(device)
model.film_layers.to(device)
model.eval()
print("Model loaded successfully.")

def load_audio(path, target_sr=8000):
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav

gul_interference = load_audio("model_test_3/Gul_01583.wav")
hassan_target    = load_audio("model_test_3/Hassan_00319.wav")
hassan_enroll    = load_audio("model_test_3/Hassan_01806.wav")

print(f"Gul interference duration: {gul_interference.shape[-1] / 8000:.2f}s")
print(f"Hassan target duration:    {hassan_target.shape[-1] / 8000:.2f}s")
print(f"Hassan enroll duration:    {hassan_enroll.shape[-1] / 8000:.2f}s")

min_len = min(gul_interference.shape[-1], hassan_target.shape[-1])
gul_interference = gul_interference[:, :min_len]
hassan_target    = hassan_target[:, :min_len]

snr_db = 0
alpha = 10 ** (-snr_db / 20)
mixture = gul_interference + alpha * hassan_target

mixture_tensor    = mixture.unsqueeze(0).to(device)
enrollment_tensor = hassan_enroll.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(mixture_tensor, enrollment_tensor)

output = output.squeeze(0).cpu()
output = output / (output.abs().max() + 1e-8)

os.makedirs("outputs_3", exist_ok=True)

torchaudio.save("outputs_3/mixture.wav",          mixture,          8000)
torchaudio.save("outputs_3/extracted_hassan.wav", output,           8000)
torchaudio.save("outputs_3/clean_hassan.wav",     hassan_target,    8000)
torchaudio.save("outputs_3/enrollment_hassan.wav",hassan_enroll,    8000)

print("Saved files to outputs_3/")