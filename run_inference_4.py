import torch
import torchaudio
import os
import torchaudio.functional as F
from conditioned_convtasnet import ConditionedConvTasNet
from speechbrain.inference import EncoderClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ConditionedConvTasNet()
checkpoint = torch.load("best_model_finetuned.pth", map_location=device)
model.convtasnet.load_state_dict(checkpoint["convtasnet"])
model.film_layers.load_state_dict(checkpoint["film_layers"])

model.to(device)
model.eval()
print("Model loaded successfully.")

ecapa = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_ecapa",
    run_opts={"device": device}
)

def load_audio(path, target_sr=8000):
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav

test_dir = os.path.join(os.getcwd(), "model_test_4")

gul_target_path = os.path.join(test_dir, "Gul_02453.wav")
gul_enroll_path = os.path.join(test_dir, "Gul_05166.wav")
haris_path      = os.path.join(test_dir, "Haris_01498.wav")
hassan_path     = os.path.join(test_dir, "Hassan_01044.wav")

for p in [gul_target_path, gul_enroll_path, haris_path, hassan_path]:
    if not os.path.exists(p):
        print(f"Missing file: {p}")
        exit()

gul_target = load_audio(gul_target_path)
gul_enroll = load_audio(gul_enroll_path)
haris      = load_audio(haris_path)
hassan     = load_audio(hassan_path)

print(f"Gul target duration: {gul_target.shape[-1] / 8000:.2f}s")
print(f"Gul enroll duration: {gul_enroll.shape[-1] / 8000:.2f}s")
print(f"Haris duration:      {haris.shape[-1] / 8000:.2f}s")
print(f"Hassan duration:     {hassan.shape[-1] / 8000:.2f}s")

min_len = min(gul_target.shape[-1], haris.shape[-1], hassan.shape[-1])
gul_target = gul_target[:, :min_len]
haris      = haris[:, :min_len]
hassan     = hassan[:, :min_len]

mixture = gul_target + 1.0 * haris + 1.0 * hassan

with torch.no_grad():
    enroll_16k = F.resample(gul_enroll, 8000, 16000).to(device)
    speaker_embedding = ecapa.encode_batch(enroll_16k).squeeze(1)

mixture_tensor = mixture.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(mixture_tensor, embedding=speaker_embedding)

output = output.squeeze(0).cpu()

if output.abs().max() > 1e-8:
    output = output * (0.9 / output.abs().max())

os.makedirs("outputs_4", exist_ok=True)

torchaudio.save("outputs_4/mixture_3spk.wav",    mixture,    8000)
torchaudio.save("outputs_4/extracted_gul.wav",   output,     8000)
torchaudio.save("outputs_4/clean_gul.wav",       gul_target, 8000)
torchaudio.save("outputs_4/enrollment_gul.wav",  gul_enroll, 8000)

print("Saved files to outputs_4/")