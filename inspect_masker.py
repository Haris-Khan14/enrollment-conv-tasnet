from asteroid.models import ConvTasNet

model = ConvTasNet.from_pretrained("mpariente/ConvTasNet_WHAM_sepclean")
print(model.masker)