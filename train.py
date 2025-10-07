import os
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torch.optim as optim
from torchvision import transforms
from PIL import Image

from siamese_datasets import get_dataset
from siamese_model import EmbeddingNet, SiameseNetwork
from losses import ContrastiveLoss


# ======= Safe Unified Transform =======
def safe_transform(x):
    """
    Converts any input to a 3x64x64 tensor:
    - PIL Image → RGB → resize → tensor
    - Tensor (grayscale 1xHxW) → repeat channels → resize → tensor
    """
    if isinstance(x, torch.Tensor):
        if x.shape[0] == 1:  # grayscale → RGB
            x = x.repeat(3, 1, 1)
        x = transforms.ToPILImage()(x)
    else:
        if x.mode != "RGB":
            x = x.convert("RGB")

    x = transforms.Resize((64, 64))(x)  # smaller for 8GB GPU
    x = transforms.ToTensor()(x)
    return x


unified_transform = transforms.Lambda(safe_transform)


# ======= Transform Wrapper =======
class TransformWrapper(torch.utils.data.Dataset):
    """
    Wrap a dataset to apply a transform at runtime
    """
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img1, img2, label = self.dataset[index]
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        label = label.float()
        return img1, img2, label

    def __len__(self):
        return len(self.dataset)


# ======= Combine All Datasets =======
def get_unified_datasets(subset_size=None):
    datasets_list = ["mnist", "fmnist", "omniglot", "celeba"]
    all_datasets = []

    for name in datasets_list:
        try:
            dataset = get_dataset(name, train=True)
            if subset_size:  # take only a subset for testing
                dataset = Subset(dataset, list(range(min(subset_size, len(dataset)))))
            dataset = TransformWrapper(dataset, unified_transform)
            all_datasets.append(dataset)
            print(f"✅ Loaded {name.upper()} with {len(dataset)} samples.")
        except Exception as e:
            print(f"⚠️ Skipping {name.upper()} due to error: {e}")

    combined = ConcatDataset(all_datasets)
    print(f"\n📦 Total combined dataset size: {len(combined)} samples\n")
    return combined


# ======= Training Function =======
def train(model, train_loader, criterion, optimizer, device, num_epochs=5, save_path="models/siamese_unified_all.pth"):
    model.train()

    # ====== Verify model works with one batch ======
    img1, img2, label = next(iter(train_loader))
    img1, img2 = img1.to(device), img2.to(device)
    try:
        out1, out2 = model(img1, img2)
        print(f"✅ Model verification successful! Embedding size: {out1.shape[1]}")
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return

    # ====== Start Training ======
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, (img1, img2, label) in enumerate(train_loader):
            if i < 5:
                print(f"Processing batch {i+1}/{len(train_loader)}")

            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"📘 Epoch [{epoch+1}/{num_epochs}] - Combined Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), save_path)
        print(f"💾 Model saved at: {save_path}")

    print("\n🎉 Training completed successfully on all datasets!\n")


# ======= Main Script =======
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    combined_dataset = get_unified_datasets(subset_size=5000)
    train_loader = DataLoader(
        combined_dataset,
        batch_size=8,         # small batch size for 8GB GPU
        shuffle=True,
        num_workers=0,        # Windows-safe
        pin_memory=True
    )

    embedding_net = EmbeddingNet()
    model = SiameseNetwork(embedding_net).to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, criterion, optimizer, device, num_epochs=5, save_path="models/siamese_unified_all.pth")
