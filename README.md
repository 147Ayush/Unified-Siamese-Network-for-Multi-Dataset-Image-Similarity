
# Siamese Network for Image/Face Verification

🎯 **Project**: Siamese Neural Network (PyTorch) — image/face verification (training, testing, visualization)  
🧑‍💻 **Author**: Ayush — *Secrets of Existence*  
📁 **Repository**: `siamese_project/`

---

## 🚀 Quick Summary

This project implements a **Siamese network** that learns embeddings for images and compares pairs using a distance metric (Euclidean). The repo contains dataset helpers, model definitions, training & evaluation scripts, and visualization utilities. A future Streamlit GUI will let users upload images and verify similarity interactively.

---

## ✅ Highlights / Features

- Train a Siamese network on several datasets (MNIST, Fashion-MNIST, Omniglot, local CelebA).
- Contrastive loss implementation for pairwise learning.
- Flexible dataset loader that builds positive/negative pairs on-the-fly.
- Safe transforms that convert grayscale→RGB and resize images for consistent input sizes.
- Unified training script that can combine multiple datasets with runtime transforms.
- Evaluation script: compute distances, ROC-AUC, distance histograms, and sample visualizations.
- GPU/CPU compatible; lightweight defaults for 8 GB GPUs.
- Modular design to swap backbones (ResNet, EfficientNet) and loss functions later.

---

## 📂 File structure (explained)

```
siamese_project/
├─ celeba_dataset/
├─ data/
├─ models/
├─ __init__.py
├─ check_dataset_shapes.py
├─ losses.py
├─ siamese_datasets.py
├─ siamese_model.py
├─ train.py
├─ test_siamese.py
├─ visualize_dataset.py
├─ requirements.txt
```

Below is a **detailed explanation** of each important `.py` file and the responsibilities inside them.

---

### `check_dataset_shapes.py`
**Purpose:** Quick script to scan a dataset directory or dataloader and verify image shapes, number of channels, and that preprocessing/resizing will produce the expected tensor shape.  
**Why use it:** Prevents shape mismatch errors before training begins. Useful when mixing datasets with different image sizes (e.g., MNIST grayscale vs. CelebA RGB).

**What it checks (recommended):**
- Number of channels (should be 3 for RGB models).
- Pixel dimensions (e.g., 128×128 or 64×64 depending on model config).
- Whether images can be opened with PIL (corrupt image detection).

**How to run:**
```bash
python check_dataset_shapes.py
```

---

### `losses.py`
**Purpose:** Implements loss functions used during training. The repo currently offers a `ContrastiveLoss` implementation.  
**Key implementation details:**
```python
L = (1 - Y) * D^2 + Y * max(0, margin - D)^2
# where Y == 0 → similar, Y == 1 → dissimilar
```
**Important note (CRITICAL):** The dataset classes in this repo sometimes use `label=1.0` for *similar* pairs (e.g., `SiameseMNIST`, `SiameseOmniglot`), while `ContrastiveLoss` **expects** `0` for similar and `1` for dissimilar — **this mismatch causes training to learn the opposite objective**. Make sure to unify the label convention across dataset and loss (recommended: `0 = similar`, `1 = dissimilar`) or flip labels in the dataset classes / loss accordingly.

**Suggested improvements:**
- Add `TripletLoss` and `BatchHardTripletLoss` as alternatives.
- Add an option to normalize embeddings (e.g. `F.normalize`) and use cosine similarity loss.

---

### `siamese_datasets.py`
**Purpose:** Dataset factory and dataset classes that produce pairs of images with labels for siamese learning. Provided classes include:
- `SiameseMNIST` (works for MNIST and FashionMNIST)
- `SiameseOmniglot`
- `SiameseCelebA` (local-folder version — reads CSVs for partitions/attributes)

**Key behavior:**
- Each `__getitem__` returns `(img1_tensor, img2_tensor, label_tensor)`.
- Pairs are generated on-the-fly: randomly decide positive/negative pair with `random.random() < 0.5`.
- `COMMON_TRANSFORM` converts grayscale→RGB, resizes (128×128 by default) and converts to `Tensor`.

**Potential issues to be aware of:**
1. **Label convention mismatch** with `ContrastiveLoss` (see `losses.py`). Fix by ensuring datasets use `0=similar` / `1=dissimilar`. Example next step: change `label = 1.0` to `label = 0.0` for same-class pairs, and vice versa.  
2. **CelebA pairing** in the current local version selects an arbitrary image as `same` without using identity metadata — CelebA attributes CSV does not directly encode identity grouping by default. If you want true identity-based positive pairs (same person), you must use the CelebA identity file (`identity_CelebA.txt`) or `list_identity.txt` (if available), map `image → person_id`, and draw positives from same person_id. Otherwise the "same" label might just be random.
3. **Memory usage**: If you convert the entire dataset to a list or load many images at once, memory can blow up. The given classes mostly load image on demand (PIL open per access) so that's OK.

**Recommended improvements:**
- Add deterministic `RandomState` or `seed` parameter for reproducible pair generation.
- Add `online_pairing` strategies and `hard_negative_sampling`.
- Support caching thumbnails for fast I/O on large datasets (optional).

---

### `siamese_model.py`
**Purpose:** Contains the CNN backbone (EmbeddingNet) and `SiameseNetwork` wrapper that sends two images through the same embedding network.  
**Two versions exist in the repo:** one lightweight dynamic `EmbeddingNet` in training script (attempting to infer FC size dynamically) and a statically-defined `EmbeddingNet` in the evaluation script. **Prefer the static one** — dynamic re-assignment of `self.fc1` inside `forward` is a fragile hack that can hide bugs and interfere with GPU placement and `model.to(device)` behavior.

**Key points:**
- Embedding output dimension: 256 (final `fc2` layer).
- Convolutional stacks reduce spatial dimension; ensure flattening size matches your input resolution (64×64 vs 128×128). If you change input size, recompute `fc1` in_features accordingly.
- After training, embeddings can be saved and used with nearest-neighbor search (Faiss), thresholding with ROC, or downstream classifiers.

**Suggested upgrades:**
- Replace the small custom conv-net with a pre-trained backbone (ResNet18/34, EfficientNet) + projection MLP (256-D).
- Add batch normalization, dropout, and embedding normalization (`L2`).
- Add a parameter to choose output embedding size easily.

---

### `train.py` (unified training script)
**Purpose:** Loads multiple datasets (MNIST/Fashion/Omniglot/CelebA), applies a *safe* unified transform that ensures all inputs become `3x64x64` tensors, wraps datasets and concatenates them, and runs contrastive training saving `models/siamese_unified_all.pth`.

**Important behavior & tips:**
- `safe_transform` converts grayscale to RGB and resizes to `64×64` (default in training script).
- Small batch size of `8` recommended for 8GB GPUs; change `batch_size` to fill GPU memory accordingly.
- `num_workers=0` is Windows-safe; on Linux increase to 4-8 for performance, but monitor memory/load.
- Before training, the script tries one forward pass to validate the model shape. If that fails, check data shape and model fc layers.

**Known mismatch caution:** The training script uses `EmbeddingNet` with a dynamic `fc1` placeholder in other parts of the repo — this causes shape confusion when switching between different versions of `EmbeddingNet`. Prefer a single canonical model definition (the static one in eval script is recommended) and keep `input_size` consistent across training and evaluation.

**How to run:**
```bash
python train.py
# or customize
python train.py --batch-size 16 --epochs 20 --dataset mnist
```

(You can easily add `argparse` flags to `train.py` for CLI configuration.)

---

### `test_siamese.py` (evaluation script)
**Purpose:** Loads a saved model and evaluates it across datasets. Computes:
- Euclidean distance between embeddings for pairs.
- ROC curve and AUC.
- Distance histograms (similar vs dissimilar pairs).
- Visualizes a few example pairs with distances on screen.

**What to pay attention to:**
- Make sure `test_siamese.py` defines the **same network architecture and input transform** that you used in training (embedding dimension, conv layers, input resize). Otherwise the `state_dict` will not match or embeddings will behave poorly.
- Use `map_location=device` when loading checkpoint if GPU types differ between train/eval.

**Run:**
```bash
python test_siamese.py
```

---

### `visualize_dataset.py`
**Purpose:** Utility script to inspect random pairs, visualize augmentations, and sanity-check transformations. Helps validate that positive / negative pairing strategy works as expected before training.

**Useful for:**
- Inspecting outputs of `safe_transform` or `COMMON_TRANSFORM`.
- Verifying color channels and image quality after preprocessing.
- Creating notebooks for qualitative analysis of model failures.

---

## 🧰 Requirements & Setup

Install dependencies (recommended in a virtual environment):

```bash
pip install -r requirements.txt
# or (example)
pip install torch torchvision numpy pillow matplotlib scikit-learn tqdm
```

**Recommended versions** (matching the project):  
- `torch==2.2.0+cu118`, `torchvision==0.17.0+cu118` (if you have CUDA 11.8)  
- `numpy==1.25.2`  
- `Pillow==10.1.0`  
- `opencv-python==4.9.0.72`  
- `tqdm`, `scikit-learn`, `matplotlib`

**GPU tips:** Use mixed precision (`torch.cuda.amp`) for speed and memory when using larger backbones. If you see OOM errors, reduce `batch_size` and/or use smaller input resolution (e.g., 64×64).

---

## 🧭 Suggested Workflows

**Train unified model (default):**
```bash
python train.py
# outputs model at: models/siamese_unified_all.pth
```

**Evaluate a trained model:**
```bash
python test_siamese.py
```
**Visualize dataset pairs:**
```bash
python visualize_dataset.py
```

**Debug shapes quickly:**
```bash
python check_dataset_shapes.py
```

---

## 🔧 Recommended Fixes & Improvements (Immediate)

1. **Unify label convention**: Make `0 = similar`, `1 = dissimilar` across datasets and the `ContrastiveLoss`. Alternatively, flip the loss formula or dataset labels. This is critical.
2. **Use a single canonical `EmbeddingNet`** (prefer static `fc1` with precomputed flatten size) to avoid dynamic redefinition inside `forward`.
3. **Make transforms consistent**: Choose either `64×64` or `128×128` across training/evaluation/visualization. Keep one config file / CLI flag for `IMG_SIZE`.
4. **CelebA identity mapping**: For face verification, generate positive pairs from the same identity using CelebA identity files, not random images. This yields correct person-level supervision.

---

## 🌟 Future roadmap & upgrades (attractive features)

- 🔁 **Streamlit web UI**: Upload two images → get similarity score and embedding visualization. Live webcam demo support.  
- 🧭 **Pretrained backbones**: Swap to ResNet/EfficientNet & fine-tune for faster convergence and better accuracy.  
- 🎯 **Online hard negative mining**: Improve embedding separation by mining difficult negatives in each batch.  
- 🧪 **Triplet and Proxy losses**: Add margin-based triplet loss, Proxy-NCA and ArcFace-style losses for face recognition quality.  
- 🗺️ **Indexing & search**: Use Faiss to index embeddings and provide fast nearest-neighbor lookup for large galleries.  
- 🚀 **Deployment**: Containerize with Docker and serve via a lightweight FastAPI backend or Streamlit Cloud for demos.  
- 📊 **Monitoring**: Add training dashboards (Weights & Biases / TensorBoard) and automated evaluation scripts.  
- 🔐 **Privacy & Security**: Add face-blurring for visualizations and opt-in consent for shared data.

---

## 🛠 Troubleshooting (common issues)

- `RuntimeError: size mismatch` when `load_state_dict`: Verify identical `EmbeddingNet` definitions and input sizes.  
- `CUDA out of memory`: lower batch size, reduce input size, or enable `torch.cuda.amp`.  
- `PIL.UnidentifiedImageError`: dataset contains corrupted images — add a try/except when opening images and skip bad files.  
- `ROC AUC unexpectedly low`: check that `labels` mapping is correct (0/1 matching similar/dissimilar) and that evaluation transform matches training transform.

---

## 🧾 License & Contact

**License:** MIT (see `LICENSE` in repo)  
**Contact:** Ayush — `ayushsaraf200@gmail.com`  

---

## 📎 Example: Quick code snippet (verify 2 images from saved model)

```python
from siamese_model import SiameseNetwork, EmbeddingNet
from siamese_datasets import safe_transform
import torch

# load model
embedding_net = EmbeddingNet()
model = SiameseNetwork(embedding_net)
model.load_state_dict(torch.load("models/siamese_unified_all.pth", map_location="cpu"))
model.eval()

# prepare images
img1 = safe_transform(Image.open("data/img1.jpg"))
img2 = safe_transform(Image.open("data/img2.jpg"))

with torch.no_grad():
    out1, out2 = model(img1.unsqueeze(0), img2.unsqueeze(0))
    dist = torch.nn.functional.pairwise_distance(out1, out2)
    print("Euclidean distance:", dist.item())
```

```
## 🚧 Current Phase (Work in Progress)

This project is currently in the **training and evaluation phase**.  
Model performance and embedding behavior are actively being tested across different datasets.  
Further tuning of **margin**, **batch size**, and **learning rate** is ongoing to improve **cross-domain generalization**.

---

## 📈 Current Progress Snapshot

- ✅ **Dataset loading and preprocessing**
- ✅ **Model architecture design**
- ✅ **Training pipeline with loss computation**
- ✅ **Evaluation and ROC visualization**
- 🚧 **Cross-domain testing** *(in progress)*
- 🚧 **Streamlit web app** *(planned)*

---

## 💡 Summary

The **Unified Siamese Network** is a flexible, research-friendly framework designed to explore  
**visual similarity learning** across multiple domains.  

It’s perfect for applications like:
- 🧍‍♂️ **Face verification**
- 👕 **Product or fashion item matching**
- ✍️ **Handwriting comparison**
- 🖼️ Any other **pairwise image similarity** task

---


