import torch
import torchaudio
from speechbrain.inference import EncoderClassifier

# Load pretrained ECAPA
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_ecapa"
)

# Load a sample enrollment segment (3 sec, 8kHz)
waveform, sr = torchaudio.load("dataset_split/train/Haris/Haris_00001.wav")

# SpeechBrain expects 16kHz
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

# Extract embedding
embedding = classifier.encode_batch(waveform)

print("Embedding shape:", embedding.shape)