import os
import random
import torch
import torchaudio
import torch.nn.functional as F


class SpeakerExtractionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, sample_rate=8000, segment_len=24000):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.segment_len = segment_len  # 3 seconds @ 8kHz

        self.speakers = [
            s for s in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, s))
        ]

        self.data = {}

        for spk in self.speakers:
            spk_path = os.path.join(root_dir, spk)
            files = [
                os.path.join(spk_path, f)
                for f in os.listdir(spk_path)
                if f.endswith(".wav")
            ]
            self.data[spk] = files

        self.spk_list = list(self.data.keys())
        self.length = sum(len(v) for v in self.data.values())

    def __len__(self):
        return self.length

    def load_audio(self, path):
        wav, sr = torchaudio.load(path)

        # Force mono
        wav = wav[:1, :]

        # Resample if needed
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        return wav

    def fix_length(self, wav):
        """Ensure waveform is exactly segment_len samples."""
        if wav.shape[-1] > self.segment_len:
            wav = wav[:, :self.segment_len]
        elif wav.shape[-1] < self.segment_len:
            pad = self.segment_len - wav.shape[-1]
            wav = F.pad(wav, (0, pad))
        return wav

    def __getitem__(self, idx):

        # -------- 1️⃣ Target Speaker --------
        target_spk = random.choice(self.spk_list)
        target_file = random.choice(self.data[target_spk])
        target = self.load_audio(target_file)
        target = self.fix_length(target)

        # -------- 2️⃣ Interferer --------
        other_spk = random.choice(
            [s for s in self.spk_list if s != target_spk]
        )
        interferer_file = random.choice(self.data[other_spk])
        interferer = self.load_audio(interferer_file)
        interferer = self.fix_length(interferer)

        # -------- 3️⃣ Random SNR Mixing --------
        snr_db = random.uniform(-5, 5)
        alpha = 10 ** (-snr_db / 20)

        mixture = target + alpha * interferer

        # -------- 4️⃣ Enrollment (Different file same speaker) --------
        enroll_file = random.choice(self.data[target_spk])
        while enroll_file == target_file:
            enroll_file = random.choice(self.data[target_spk])

        enrollment = self.load_audio(enroll_file)
        enrollment = self.fix_length(enrollment)

        return mixture, target, enrollment