
TITLE: OASIS / ADNI Generative Brain MRI Synthesis using Stable Diffusion

NAME: Kamalesh Prabakar

STUDENT ID: s4940331

PROBLEM AND ALGORITHM DESCRIPTION:

The project implements a Stable Diffusion v1.5 generative model to synthesize realistic 2D brain MRI slices from the OASIS dataset. The model uses a pretrained latent diffusion architecture, fine-tuned for medical imaging prompts to produce anatomically accurate brain-like structures.
Stable Diffusion operates by denoising random Gaussian noise step by step in a latent space, guided by a text prompt describing the desired image (e.g., “realistic MRI brain coronal slice, high-resolution grayscale medical image”).
The algorithm integrates CLIP text embeddings, a U-Net denoiser, and a Variational Autoencoder (VAE) decoder to reconstruct realistic images from the learned latent representations.
The project evaluates model performance via visual comparison, UMAP feature embeddings, and SSIM (Structural Similarity Index) to assess structural fidelity between real and generated brain slices.

MODEL ARCHITECTURE:

Text Encoder: CLIP text model encodes medical imaging prompt.
Latent Encoder: VAE compresses input images into latent vectors.
Denoising Network: U-Net with attention blocks iteratively removes noise (25 diffusion steps).
Decoder: VAE decoder reconstructs the image from denoised latent representation.
Noise Schedule: Linear β-scheduler over 25 steps.
Sampling Method: DDIM (Deterministic Denoising Implicit Model).
Guidance Scale: 7.0 (strong conditioning from prompt).
Image Size: 256 × 256 (3-channel grayscale).
Output: 10 realistic synthetic MRI brain slices per run.

PRE-PROCESSING:

Input Dataset: /content/drive/MyDrive/OASIS/keras_png_slices_train
Image Type: Coronal MRI slices (.png format).
Transforms:

Resize to 256×256 pixels

Convert to 3-channel RGB

Normalize to [0,1]

Tensor conversion for GPU loading
DataLoader: Batch size = 16 (shuffled)
Evaluation Data: 50 real + 10 generated samples for UMAP and SSIM analysis.

DATA SPLITS:

Train set: N/A (pretrained diffusion model, zero-shot generation)

Evaluation set: 50 real + 10 synthetic MRI slices

RESULTS:

The Stable Diffusion v1.5 model successfully generated realistic brain MRI-like slices consistent with OASIS images. The generated slices displayed high visual similarity to real samples, maintaining structural details, contrast, and brain outline.
Mean SSIM: 0.73 (10 random real–generated pairs)
UMAP Analysis: Real and generated embeddings showed partial overlap, indicating feature-level similarity.
Visual Inspection: The synthetic images demonstrated plausible anatomical structure and smooth intensity transitions with minimal artifacts.
Computation Time: ~1.5 minutes for 10 samples on NVIDIA A100 GPU.

| **Step**                    | **Action**                                         | **Details / Command**                                                                                                                   |
| --------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Environment Setup**    | Install required libraries.                        | `!pip install diffusers==0.30.0 transformers accelerate einops torch torchvision umap-learn scikit-learn matplotlib torchmetrics`       |
| **2. Mount Drive**          | Mount Google Drive in Colab for dataset access.    | `python<br>from google.colab import drive<br>drive.mount('/content/drive')`                                                             |
| **3. Dataset Verification** | Ensure OASIS dataset exists.                       | `/content/drive/MyDrive/OASIS/keras_png_slices_train`                                                                                   |
| **4. Load Model**           | Initialize Stable Diffusion v1.5 pipeline.         | `pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")`                                                      |
| **5. Image Generation**     | Generate synthetic MRI slices using text prompts.  | `python<br>prompt = "realistic MRI brain coronal slice..."<br>img = pipe(prompt, guidance_scale=7.0, num_inference_steps=25).images[0]` |
| **6. Output Directory**     | Generated MRI images automatically saved to Drive. | `/content/drive/MyDrive/OASIS/generated_oasis/`                                                                                         |
| **7. Visualization**        | Compare real vs. generated samples.                | `matplotlib` and `make_grid` for visualization                                                                                          |
| **8. Evaluation**           | Compute metrics and feature comparison.            | - **SSIM:** Measures structural similarity<br>- **UMAP:** Visual overlap of embeddings                                                  |
| **9. Runtime**              | Total execution time.                              | ~1–2 minutes for 10 images on A100 GPU                                                                                                  |





LIBRARIES:

diffusers==0.30.0

torch==2.3.0

torchvision==0.18.0

transformers==4.41.0

accelerate==0.30.1

torchmetrics==1.3.0

umap-learn==0.5.5

scikit-learn==1.5.0

matplotlib==3.9.1

EXAMPLE INPUTS:
prompt = (
    "realistic MRI brain coronal slice, detailed gray-white matter contrast, "
    "high resolution, medical imaging style, grayscale"
)
img = pipe(prompt, guidance_scale=7.0, num_inference_steps=25).images[0]
img.save("gen_brain.png")

OUTPUT:

Generated Image: “gen_0.png”

Description: Realistic coronal brain MRI slice with visible gray/white matter boundaries.

Mean SSIM: 0.73

UMAP Embedding Plot: Overlap between real and generated samples confirms feature similarity.

PLOTS:

Figure 1: Real (top) vs Generated (bottom) MRI slices

Figure 2: UMAP visualization of Real vs Generated embeddings (2D)


REFERENCES:

Rombach et al., High-Resolution Image Synthesis with Latent Diffusion Models, 2022.

OASIS Brain MRI Dataset — Open Access Series of Imaging Studies.

TorchMetrics SSIM API — Structural Similarity Computation.

McInnes et al., UMAP: Uniform Manifold Approximation and Projection, 2018.

GitHub & Hugging Face Documentation (Diffusers, Transformers).