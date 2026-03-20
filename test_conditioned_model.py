import torch
from conditioned_convtasnet import ConditionedConvTasNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConditionedConvTasNet()

# Move only separator part to GPU
model.convtasnet.to(device)
model.film_layers.to(device)

model.eval()

mixture = torch.randn(2, 1, 24000).to(device)
enrollment = torch.randn(2, 1, 24000)   # keep on CPU

with torch.no_grad():
    out = model(mixture, enrollment)

print("Output shape:", out.shape)

if torch.cuda.is_available():
    print("GPU memory allocated:",
          torch.cuda.memory_allocated() / 1024**2, "MB")