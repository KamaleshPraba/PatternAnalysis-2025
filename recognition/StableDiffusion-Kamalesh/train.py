# train.py
import os, random, torch, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision.utils import make_grid
from torchvision.models import resnet18
from sklearn.preprocessing import StandardScaler
import umap
import skimage.metrics as metrics

from dataset import get_dataloader
from modules import load_diffusion_model

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_images(pipe, output_dir, prompt, n=10):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(n)):
        img = pipe(prompt, guidance_scale=7.0, num_inference_steps=25).images[0]
        img.save(output_dir / f"gen_{i}.png")
    print(f"✅ Generated {n} samples → {output_dir}")

def visualize_samples(real_dir, gen_dir, transform):
    real_imgs = [transform(Image.open(p)) for p in list(Path(real_dir).glob('*.png'))[:4]]
    gen_imgs  = [transform(Image.open(p)) for p in list(Path(gen_dir).glob('*.png'))[:4]]
    grid = make_grid(real_imgs + gen_imgs, nrow=4)
    plt.figure(figsize=(8,4))
    plt.imshow(np.transpose(grid, (1,2,0)))
    plt.axis('off')
    plt.title("Top: Real | Bottom: Generated")
    plt.show()

def compute_umap(real_dir, gen_dir, transform):
    encoder = resnet18(pretrained=True).to(device)
    encoder.fc = torch.nn.Identity()

    def get_emb(img_dir, limit=50):
        imgs = [transform(Image.open(p).convert("RGB")).unsqueeze(0)
                for p in list(Path(img_dir).glob("*.png"))[:limit]]
        imgs = torch.cat(imgs).to(device)
        with torch.no_grad():
            return encoder(imgs).cpu().numpy()

    real_emb, gen_emb = get_emb(real_dir), get_emb(gen_dir)
    all_emb = np.vstack([real_emb, gen_emb])
    labels  = np.array([0]*len(real_emb) + [1]*len(gen_emb))
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42)
    emb2d   = reducer.fit_transform(StandardScaler().fit_transform(all_emb))

    plt.figure(figsize=(6,5))
    plt.scatter(emb2d[labels==0,0], emb2d[labels==0,1], c='royalblue', s=10, label='Real')
    plt.scatter(emb2d[labels==1,0], emb2d[labels==1,1], c='tomato', s=10, label='Generated')
    plt.title("UMAP Embeddings: Real vs Generated")
    plt.legend()
    plt.show()

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
    return np.mean(scores)
