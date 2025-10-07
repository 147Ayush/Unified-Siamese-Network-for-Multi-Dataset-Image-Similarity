import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from siamese_datasets import get_dataset
from siamese_model import SiameseNetwork


# ============================================================
# ✅ Embedding Network (matches the 64x64 model used in training)
# ============================================================
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),  # (3,64,64) -> (32,60,60)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),               # (32,30,30)

            nn.Conv2d(32, 64, kernel_size=5), # (64,26,26)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),               # (64,13,13)

            nn.Conv2d(64, 128, kernel_size=3),# (128,11,11)
            nn.ReLU(inplace=True)
        )

        # Flatten dimension for 64x64 input = 128*11*11 = 15488
        self.fc1 = nn.Linear(128 * 11 * 11, 512)
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================
# ✅ Safe Transform (Convert to 3x64x64 Tensor)
# ============================================================
def safe_transform(x):
    if isinstance(x, torch.Tensor):
        if x.shape[0] == 1:  # grayscale → RGB
            x = x.repeat(3, 1, 1)
        x = transforms.ToPILImage()(x)
    else:
        if x.mode != "RGB":
            x = x.convert("RGB")
    x = transforms.Resize((64, 64))(x)
    x = transforms.ToTensor()(x)
    return x


# ============================================================
# ✅ Verification Functions
# ============================================================
def verify_image_shape(img, expected_shape=(3, 64, 64)):
    assert img.shape == expected_shape, f"❌ Image shape mismatch: got {img.shape}, expected {expected_shape}"
    print(f"✅ Image shape verified: {img.shape}")

def verify_embedding_shape(model, device):
    model.eval()
    dummy = torch.randn(1, 3, 64, 64).to(device)
    out = model.embedding_net(dummy)
    assert out.shape[-1] == 256, f"❌ Embedding output mismatch: got {out.shape}"
    print(f"✅ Embedding shape verified: {out.shape}")


# ============================================================
# ✅ Evaluation per Dataset
# ============================================================
def evaluate_dataset(model, dataset_name, device, max_samples=2000, visualize_samples=3):
    print(f"\n🧪 Evaluating on dataset: {dataset_name.upper()}")

    try:
        dataset = get_dataset(dataset_name, train=False)
    except Exception as e:
        print(f"⚠️ Skipping {dataset_name} due to error: {e}")
        return

    transform = transforms.Lambda(safe_transform)

    def clean_label(c):
        if torch.is_tensor(c):
            return c.detach().clone().float()
        else:
            return torch.tensor(c, dtype=torch.float32)

    dataset = [(transform(a), transform(b), clean_label(c)) for a, b, c in dataset]
    subset = Subset(dataset, range(0, min(max_samples, len(dataset))))
    dataloader = DataLoader(subset, batch_size=8, shuffle=False, num_workers=0)

    img1, img2, label = next(iter(dataloader))
    verify_image_shape(img1[0])
    verify_embedding_shape(model, device)

    distances, labels = [], []

    with torch.no_grad():
        for i, (img1, img2, label) in enumerate(dataloader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            out1, out2 = model(img1, img2)
            euclidean_distance = F.pairwise_distance(out1, out2)
            distances.extend(euclidean_distance.cpu().numpy())
            labels.extend(label.cpu().numpy())

            if i < visualize_samples:
                visualize_pair(img1[0], img2[0], label[0].item(), euclidean_distance[0].item())

    distances = np.array(distances)
    labels = np.array(labels)

    fpr, tpr, _ = roc_curve(labels, distances)
    roc_auc = auc(fpr, tpr)
    print(f"🔹 {dataset_name.upper()} ROC AUC Score: {roc_auc:.4f}")

    plot_distance_histogram(distances, labels, dataset_name)
    plot_roc_curve(fpr, tpr, roc_auc, dataset_name)


# ============================================================
# ✅ Visualization Helpers
# ============================================================
def visualize_pair(img1, img2, label, distance):
    img1_np = img1.cpu().permute(1, 2, 0).numpy()
    img2_np = img2.cpu().permute(1, 2, 0).numpy()

    plt.figure(figsize=(3, 2))
    plt.subplot(1, 2, 1)
    plt.imshow(img1_np)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img2_np)
    plt.axis("off")

    plt.suptitle(f"Label: {int(label)} | Distance: {distance:.4f}")
    plt.show()


def plot_distance_histogram(distances, labels, dataset_name):
    plt.figure(figsize=(6, 4))
    plt.hist(distances[labels == 0], bins=30, alpha=0.6, label="Similar (Label=0)")
    plt.hist(distances[labels == 1], bins=30, alpha=0.6, label="Dissimilar (Label=1)")
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Count")
    plt.title(f"{dataset_name.upper()} - Distance Distribution")
    plt.legend()
    plt.show()


def plot_roc_curve(fpr, tpr, roc_auc, dataset_name):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{dataset_name.upper()} - ROC Curve")
    plt.legend(loc="lower right")
    plt.show()


# ============================================================
# ✅ Main Script
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ===== Load trained model =====
    embedding_net = EmbeddingNet()
    model = SiameseNetwork(embedding_net)
    model.load_state_dict(torch.load("models/siamese_unified_all.pth", map_location=device))
    model.to(device)

    # ===== Evaluate across datasets =====
    datasets = ["mnist", "fmnist", "omniglot", "celeba"]
    for name in datasets:
        evaluate_dataset(model, name, device, max_samples=2000, visualize_samples=2)

    print("\n🎉 All dataset evaluations completed successfully!")
