# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image

class OASISDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = list(Path(root).glob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0

def get_dataloader(data_root, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])
    dataset = OASISDataset(data_root, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"✅ Loaded {len(dataset)} OASIS PNG slices")
    return loader
