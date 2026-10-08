"""Turn stored raw worlds into a separate packed token artifact."""

from nca.dataset import TrajectoryDataset
from sampleeff_nca_pretraining.packing import pack_dataset


raw = TrajectoryDataset.load("runs/tiny/raw")
packed = pack_dataset(raw)
print("token shape:", tuple(packed.sequences.shape))
print(packed.save("runs/tiny/packed.pt"))
