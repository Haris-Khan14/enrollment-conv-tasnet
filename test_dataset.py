from Dataset_class import SpeakerExtractionDataset
from torch.utils.data import DataLoader

dataset = SpeakerExtractionDataset("dataset_split/train")

loader = DataLoader(dataset, batch_size=2)

mixture, target, enrollment = next(iter(loader))

print("Mixture shape:", mixture.shape)
print("Target shape:", target.shape)
print("Enrollment shape:", enrollment.shape)