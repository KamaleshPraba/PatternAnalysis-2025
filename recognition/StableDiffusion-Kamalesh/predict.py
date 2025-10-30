# predict.py
import torch
from pathlib import Path
from torchvision import transforms
from dataset import get_dataloader
from modules import load_diffusion_model
from train import generate_images, visualize_samples, compute_umap, avg_ssim

device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path("/content/drive/MyDrive/OASIS/keras_png_slices_train")
OUTPUT_DIR = Path("/content/drive/MyDrive/OASIS/generated_oasis")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# 1️⃣ Load data + model
loader = get_dataloader(DATA_ROOT)
pipe = load_diffusion_model(device=device)

# 2️⃣ Generate synthetic images
prompt = ("realistic MRI brain coronal slice, detailed gray-white matter contrast, "
          "high resolution, medical imaging style, grayscale")
generate_images(pipe, OUTPUT_DIR, prompt, n=10)

# 3️⃣ Visual checks and metrics
visualize_samples(DATA_ROOT, OUTPUT_DIR, transform)
compute_umap(DATA_ROOT, OUTPUT_DIR, transform)
mean_ssim = avg_ssim(DATA_ROOT, OUTPUT_DIR)
print(f"✅ Mean SSIM: {mean_ssim:.3f}")


import glob
from IPython.display import Image as ColabImage, display

# Display generated images
gen_images = sorted(glob.glob(str(output_dir / "*.png")))
print(f"🧠 Showing {len(gen_images)} generated samples:")
for p in gen_images[:5]:   # show first 5 only
    display(ColabImage(p))

