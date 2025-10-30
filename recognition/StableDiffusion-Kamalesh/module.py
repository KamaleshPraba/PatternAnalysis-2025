# ============================================================
# modules.py — Common components for OASIS/ADNI Generative Model
# ============================================================

import torch, random, numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset
import umap
from sklearn.preprocessing import StandardScaler
import skimage.metrics as metrics
import matplotlib.pyplot as plt


# --------------------------
# 🧩 OASIS Dataset
# --------------------------
class OASISDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = list(Path(root).glob("*.png"))
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, 0


# --------------------------
# ⚙️ Transform setup
# --------------------------
def get_default_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])


# --------------------------
# 🧠 ResNet18 Feature Encoder
# --------------------------
def get_resnet_encoder(device="cpu"):
    encoder = models.resnet18(pretrained=True).to(device)
    encoder.fc = torch.nn.Identity()
    return encoder


# --------------------------
# 🔍 UMAP Embedding Function
# --------------------------
def extract_umap_embeddings(encoder, real_dir, gen_dir, transform, device="cpu"):
    def get_embeddings_from_dir(img_dir, limit=50):
        imgs = [transform(Image.open(p).convert("RGB")).unsqueeze(0)
                for p in list(Path(img_dir).glob("*.png"))[:limit]]
        if not imgs:
            return np.zeros((0,512))
        imgs = torch.cat(imgs).to(device)
        with torch.no_grad():
            return encoder(imgs).cpu().numpy()

    real_emb = get_embeddings_from_dir(real_dir)
    gen_emb  = get_embeddings_from_dir(gen_dir)
    all_emb  = np.vstack([real_emb, gen_emb])
    labels   = np.array([0]*len(real_emb) + [1]*len(gen_emb))

    reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42)
    emb2d = reducer.fit_transform(StandardScaler().fit_transform(all_emb))

    plt.figure(figsize=(6,5))
    plt.scatter(emb2d[labels==0,0], emb2d[labels==0,1], c='royalblue', s=10, label='Real')
    plt.scatter(emb2d[labels==1,0], emb2d[labels==1,1], c='tomato', s=10, label='Generated')
    plt.title("UMAP Embeddings: Real vs Generated OASIS Slices")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.legend(); plt.show()


# --------------------------
# 📊 SSIM Metric Functions
# --------------------------
def compute_ssim(img1, img2):
    img1 = np.array(img1.resize((128,128)).convert("L"))
    img2 = np.array(img2.resize((128,128)).convert("L"))
    return metrics.structural_similarity(img1, img2)

def avg_ssim(real_dir, gen_dir, n=10):
    reals = list(Path(real_dir).glob("*.png"))
    gens  = list(Path(gen_dir).glob("*.png"))
    scores = []
    for _ in range(n):
        r = Image.open(random.choice(reals))
        g = Image.open(random.choice(gens))
        scores.append(compute_ssim(g, r))
    return np.mean(scores)
