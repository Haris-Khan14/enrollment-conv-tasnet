import torch
import torch.nn as nn
from asteroid.models import ConvTasNet
from speechbrain.inference import EncoderClassifier
import torchaudio

class ConditionedConvTasNet(nn.Module):
    def __init__(self):
        super().__init__()
      
        self.convtasnet = ConvTasNet.from_pretrained("mpariente/ConvTasNet_WHAM_sepclean")

   
        self.convtasnet.masker.n_src = 1
        in_chan = self.convtasnet.masker.mask_net[1].in_channels
        out_chan = self.convtasnet.masker.out_chan
        self.convtasnet.masker.mask_net[1] = nn.Conv1d(in_chan, out_chan, 1)

      
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

    def forward(self, mixture, enrollment=None, embedding=None):
        """
        mixture: [B, 1, T] or [B, T]
        enrollment: [B, 1, T_enroll] (Audio - used for Laptop testing)
        embedding: [B, 192] (Vector - used for Pi/ONNX optimization)
        """
        if mixture.ndim == 2: 
            mixture = mixture.unsqueeze(1)

      
        if embedding is None:
            if enrollment is None:
                raise ValueError("Inference requires either 'enrollment' audio or 'embedding' vector.")
            
            if enrollment.ndim == 2: 
                enrollment = enrollment.unsqueeze(1)

            with torch.no_grad():
                enroll_cpu = enrollment.squeeze(1).cpu()
                enroll_16k = torchaudio.functional.resample(enroll_cpu, 8000, 16000)
                embedding = self.ecapa.encode_batch(enroll_16k).squeeze(1)
                embedding = embedding.to(mixture.device)

       
        enc_out = self.convtasnet.encoder(mixture)
        x = self.convtasnet.masker.bottleneck(enc_out)
        skip_connections = 0

        for i, block in enumerate(self.convtasnet.masker.TCN):
            y = block.shared_block(x)
            
        
            film_params = self.film_layers[i](embedding).unsqueeze(-1)
            gamma, beta = torch.chunk(film_params, 2, dim=1)
            y = gamma * y + beta
            
            x = x + block.res_conv(y)
            skip_connections = skip_connections + block.skip_conv(y)

        masks = self.convtasnet.masker.mask_net(skip_connections)
        masks = self.convtasnet.masker.output_act(masks)
        out = self.convtasnet.decoder(enc_out * masks)
        
        return out