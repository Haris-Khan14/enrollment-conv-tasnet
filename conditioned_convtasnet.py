import torch
import torch.nn as nn
from asteroid.models import ConvTasNet
from speechbrain.inference import EncoderClassifier
import torchaudio


class ConditionedConvTasNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pretrained ConvTasNet
        self.convtasnet = ConvTasNet.from_pretrained(
            "mpariente/ConvTasNet_WHAM_sepclean"
        )

        # ---- Modify only final projection for 1 source ----
        self.convtasnet.masker.n_src = 1
        in_chan = self.convtasnet.masker.mask_net[1].in_channels
        out_chan = self.convtasnet.masker.out_chan
        self.convtasnet.masker.mask_net[1] = nn.Conv1d(in_chan, out_chan, 1)

        # ---- FiLM layers for each TCN block ----
        self.film_layers = nn.ModuleList([
            nn.Linear(192, 512 * 2) for _ in range(len(self.convtasnet.masker.TCN))
        ])

        self.ecapa = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_ecapa",
    run_opts={"device": "cpu"}
)

        for param in self.ecapa.parameters():
            param.requires_grad = False


    def forward(self, mixture, enrollment):

        if mixture.ndim == 2:
            mixture = mixture.unsqueeze(1)
        if enrollment.ndim == 2:
            enrollment = enrollment.unsqueeze(1)

        # -------- Speaker Embedding --------
        with torch.no_grad():
            enroll_cpu = enrollment.squeeze(1).cpu()
            enroll_16k = torchaudio.functional.resample(enroll_cpu, 8000, 16000)
            emb = self.ecapa.encode_batch(enroll_16k).squeeze(1)
            emb = emb.to(mixture.device)

        # -------- Encoder --------
        enc_out = self.convtasnet.encoder(mixture)

        # -------- Bottleneck --------
        x = self.convtasnet.masker.bottleneck(enc_out)

        skip_connections = 0

        # -------- Proper TCN Loop --------
        for i, block in enumerate(self.convtasnet.masker.TCN):

            # shared_block
            y = block.shared_block(x)   # [B, 512, T]

            # FiLM injection on 512-dim feature
            film_params = self.film_layers[i](emb).unsqueeze(-1)
            gamma, beta = torch.chunk(film_params, 2, dim=1)
            y = gamma * y + beta

            # residual & skip
            res = block.res_conv(y)
            skip = block.skip_conv(y)

            x = x + res
            skip_connections = skip_connections + skip

        # -------- Final mask generation --------
        masks = self.convtasnet.masker.mask_net(skip_connections)
        masks = self.convtasnet.masker.output_act(masks)

        # -------- Decode --------
        out = self.convtasnet.decoder(enc_out * masks)

        return out