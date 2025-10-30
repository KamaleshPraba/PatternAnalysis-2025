# ============================================================
# train.py — Stable Diffusion OASIS/ADNI Image Generation
# ============================================================

import os, random, torch, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision.utils import make_grid
from torchvision.models import resnet18
from sklearn.preprocessing import StandardScaler
import umap
import skimage.metrics as metrics

from dataset import get_dataloader, get_default_transform
from modules import load_diffusion_model

# --------------------------
# 0️⃣ Setup + Config
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DATA_ROOT = Path("/content/drive/MyDrive/OASIS/keras_png_slices_train")
OUTPUT_DIR = Path("/content/drive/MyDrive/OASIS/generated_oasis")
PROMPT = (
    "realistic MRI brain coronal slice, detailed gray-white matter contrast, "
    "high resolution, medical imaging style, grayscale"
)

# ============================================================
# 🔧 1️⃣ IMAGE GENERATION
# ============================================================
def generate_images(pipe, output_dir, prompt, n=10):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(n), desc="🧠 Generating synthetic slices"):
        img = pipe(prompt, guidance_scale=7.0, num_inference_steps=25).images[0]
        img.save(output_dir / f"gen_{i}.png")
    print(f"✅ Generated {n} synthetic samples → {output_dir}")


# ============================================================
# 🖼️ 2️⃣ VISUALIZATION
# ============================================================
def visualize_samples(real_dir, gen_dir, transform):
    real_imgs = [transform(Image.open(p)) for p in list(Path(real_dir).glob('*.png'))[:4]]
    gen_imgs  = [transform(Image.open(p)) for p in list(Path(gen_dir).glob('*.png'))[:4]]

    grid = make_grid(real_imgs + gen_imgs, nrow=4)
    plt.figure(figsize=(8,4))
    plt.imshow(np.transpose(grid, (1,2,0)))
    plt.axis('off')
    plt.title("Top: Real | Bottom: Generated")
    plt.show()


# ============================================================
# 🔍 3️⃣ UMAP ANALYSIS
# ============================================================
def compute_umap(real_dir, gen_dir, transform):
    encoder = resnet18(pretrained=True).to(device)
    encoder.fc = torch.nn.Identity()

    def get_embeddings(img_dir, limit=50):
        imgs = [transform(Image.open(p).convert("RGB")).unsqueeze(0)
                for p in list(Path(img_dir).glob("*.png"))[:limit]]
        if not imgs:
            return np.zeros((0,512))
        imgs = torch.cat(imgs).to(device)
        with torch.no_grad():
            return encoder(imgs).cpu().numpy()

    real_emb, gen_emb = get_embeddings(real_dir), get_embeddings(gen_dir)
    all_emb = np.vstack([real_emb, gen_emb])
    labels  = np.array([0]*len(real_emb) + [1]*len(gen_emb))

    reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42)
    emb2d = reducer.fit_transform(StandardScaler().fit_transform(all_emb))

    plt.figure(figsize=(6,5))
    plt.scatter(emb2d[labels==0,0], emb2d[labels==0,1], c='royalblue', s=10, label='Real')
    plt.scatter(emb2d[labels==1,0], emb2d[labels==1,1], c='tomato', s=10, label='Generated')
    plt.title("UMAP: Real vs Generated OASIS Slices")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.legend()
    plt.show()


# ============================================================
# 📊 4️⃣ SSIM METRIC
# ============================================================
def avg_ssim(real_dir, gen_dir, n=10):
    reals = list(Path(real_dir).glob("*.png"))
    gens  = list(Path(gen_dir).glob("*.png"))
    scores = []
    for _ in range(n):
        r = Image.open(random.choice(reals))
        g = Image.open(random.choice(gens))
        r = np.array(r.resize((128,128)).convert("L"))
        g = np.array(g.resize((128,128)).convert("L"))
        scores.append(metrics.structural_similarity(g, r))
    mean_ssim = np.mean(scores)
    print(f"✅ Mean SSIM over {n} random pairs: {mean_ssim:.3f}")
    return mean_ssim


# ============================================================
# 🚀 5️⃣ MAIN EXECUTION PIPELINE
# ============================================================
def main():
    print(f"✅ Using device: {device}")
    assert DATA_ROOT.exists(), f"{DATA_ROOT} not found"

    transform = get_default_transform()
    dataloader = get_dataloader(DATA_ROOT, transform)
    print(f"✅ Loaded {len(dataloader.dataset)} OASIS slices")

    # Load Stable Diffusion model
    pipe = load_diffusion_model(device)

    # Generate & visualize
    generate_images(pipe, OUTPUT_DIR, PROMPT, n=10)
    visualize_samples(DATA_ROOT, OUTPUT_DIR, transform)

    # UMAP + SSIM evaluation
    compute_umap(DATA_ROOT, OUTPUT_DIR, transform)
    avg_ssim(DATA_ROOT, OUTPUT_DIR, n=10)


if __name__ == "__main__":
    main()
