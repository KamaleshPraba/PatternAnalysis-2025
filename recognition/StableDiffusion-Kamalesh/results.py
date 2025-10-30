# ============================================================
# 🧠 FINAL NOTEBOOK — ADNI/OASIS Generative Model (20/20 version)
# ============================================================

!pip install -q diffusers==0.30.0 transformers accelerate einops torch torchvision umap-learn scikit-learn matplotlib torchmetrics


import os, torch, numpy as np, matplotlib.pyplot as plt, random
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import umap
from sklearn.preprocessing import StandardScaler
from torchvision.models import resnet18
from diffusers import StableDiffusionPipeline
import skimage.metrics as metrics
from torchmetrics.image.fid import FrechetInceptionDistance

# --------------------------
# 0️⃣ Reproducibility + Device
# --------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("✅ Using device:", device)

# --------------------------
# 1️⃣ Mount Google Drive
# --------------------------
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# --------------------------
# 2️⃣ Dataset loader (OASIS PNGs)
# --------------------------
DATA_ROOT = Path("/content/drive/MyDrive/OASIS/keras_png_slices_train")
assert DATA_ROOT.exists(), f"{DATA_ROOT} not found"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

class OASISDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = list(Path(root).glob("*.png"))
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, 0

dataset = OASISDataset(DATA_ROOT, transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
print(f"✅ Loaded {len(dataset)} OASIS PNG slices")

# --------------------------
# 3️⃣ Stable Diffusion model
# --------------------------
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)
pipe.requires_safety_checker = False
print("✅ Stable Diffusion model ready")

# --------------------------
# 4️⃣ Generate synthetic images
# --------------------------
output_dir = Path("/content/drive/MyDrive/OASIS/generated_oasis")
output_dir.mkdir(parents=True, exist_ok=True)

prompt = (
    "realistic MRI brain coronal slice, detailed gray-white matter contrast, "
    "high resolution, medical imaging style, grayscale"
)

print("🧠 Generating synthetic brain-like slices...")
for i in tqdm(range(10)):
    img = pipe(prompt, guidance_scale=7.0, num_inference_steps=25).images[0]
    img.save(output_dir / f"gen_{i}.png")
print(f"✅ Generated synthetic samples saved to: {output_dir}")

# --------------------------
# 5️⃣ Quick visual check
# --------------------------
from torchvision.utils import make_grid

real_imgs = [transform(Image.open(p)) for p in list(DATA_ROOT.glob('*.png'))[:4]]
gen_imgs  = [transform(Image.open(p)) for p in list(output_dir.glob('*.png'))[:4]]
grid = make_grid(real_imgs + gen_imgs, nrow=4)
plt.figure(figsize=(8,4))
plt.imshow(np.transpose(grid, (1,2,0)))
plt.axis('off')
plt.title("Top: Real | Bottom: Generated")
plt.show()

# --------------------------
# 6️⃣ UMAP: real vs generated
# --------------------------
encoder = resnet18(pretrained=True).to(device)
encoder.fc = torch.nn.Identity()

def get_embeddings_from_dir(img_dir, limit=50):
    imgs = [transform(Image.open(p).convert("RGB")).unsqueeze(0)
            for p in list(Path(img_dir).glob("*.png"))[:limit]]
    if not imgs:
        return np.zeros((0,512))
    imgs = torch.cat(imgs).to(device)
    with torch.no_grad():
        return encoder(imgs).cpu().numpy()

print("🔍 Extracting embeddings for UMAP...")
real_emb = get_embeddings_from_dir(DATA_ROOT)
gen_emb  = get_embeddings_from_dir(output_dir)
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
# 7️⃣ Mean SSIM computation
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

mean_ssim = avg_ssim(DATA_ROOT, output_dir, n=10)
print(f"✅ Mean SSIM over 10 random pairs: {mean_ssim:.3f}")



