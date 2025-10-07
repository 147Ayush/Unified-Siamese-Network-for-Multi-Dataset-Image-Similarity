import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets import Omniglot
from PIL import Image
import pandas as pd

# Common transform for all datasets (convert to RGB, resize to 128x128)
COMMON_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale → RGB
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# ============ MNIST / Fashion-MNIST ============
class SiameseMNIST(Dataset):
    def __init__(self, train=True, fashion=False):
        self.transform = COMMON_TRANSFORM
        if fashion:
            self.dataset = datasets.FashionMNIST(root="data", train=train, download=True)
        else:
            self.dataset = datasets.MNIST(root="data", train=train, download=True)
        self.targets = self.dataset.targets.numpy()

    def __getitem__(self, index):
        img1, label1 = self.dataset[index]
        if random.random() < 0.5:
            idx2 = random.choice((self.targets == label1).nonzero()[0])
            label = 1.0
        else:
            idx2 = random.choice((self.targets != label1).nonzero()[0])
            label = 0.0
        img2, _ = self.dataset[idx2]

        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2, torch.tensor([label], dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)


# ============ Omniglot ============
class SiameseOmniglot(Dataset):
    def __init__(self, background=True):
        self.transform = COMMON_TRANSFORM
        self.dataset = Omniglot(root="data", background=background, download=True)
        self.data = [(img, label) for img, label in self.dataset]

    def __getitem__(self, index):
        img1, label1 = self.data[index]
        if random.random() < 0.5:
            candidates = [i for i, (img, l) in enumerate(self.data) if l == label1 and i != index]
            idx2 = random.choice(candidates)
            label = 1.0
        else:
            candidates = [i for i, (img, l) in enumerate(self.data) if l != label1]
            idx2 = random.choice(candidates)
            label = 0.0
        img2, _ = self.data[idx2]

        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2, torch.tensor([label], dtype=torch.float32)

    def __len__(self):
        return len(self.data)


# ============ CelebA (LOCAL VERSION) ============
class SiameseCelebA(Dataset):
    def __init__(self, root="data", split="train"):
        """
        Custom Siamese Dataset for CelebA (uses local files only)
        Folder structure must be:
            data/
            └── celeba/
                ├── img_align_celeba/
                ├── list_attr_celeba.csv
                ├── list_eval_partition.csv
        """
        self.root = os.path.join(root, "celeba")
        self.split = split
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        # Load partition info
        partition_df = pd.read_csv(os.path.join(self.root, "list_eval_partition.csv"))
        partition_df.columns = ["image_id", "partition"]
        split_map = {"train": 0, "valid": 1, "test": 2}
        split_id = split_map[split]

        # Load attributes
        attr_df = pd.read_csv(os.path.join(self.root, "list_attr_celeba.csv"))
        attr_df = attr_df.merge(partition_df, on="image_id")
        self.data = attr_df[attr_df["partition"] == split_id].reset_index(drop=True)

        self.image_dir = os.path.join(self.root, "img_align_celeba")
        self.image_list = self.data["image_id"].tolist()

        print(f"[INFO] Loaded {len(self.image_list)} CelebA images for '{split}' split (local).")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # Load first image
        img1_name = self.image_list[index]
        img1_path = os.path.join(self.image_dir, img1_name)
        img1 = Image.open(img1_path).convert("RGB")

        # Decide if same or different pair
        same_class = random.random() < 0.5

        idx2 = random.choice(range(len(self.image_list)))
        label = 1.0 if same_class else 0.0

        img2_name = self.image_list[idx2]
        img2_path = os.path.join(self.image_dir, img2_name)
        img2 = Image.open(img2_path).convert("RGB")

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, torch.tensor([label], dtype=torch.float32)


# ============ Dataset Factory ============
def get_dataset(name, train=True):
    if name == "mnist":
        return SiameseMNIST(train=train, fashion=False)
    elif name == "fmnist":
        return SiameseMNIST(train=train, fashion=True)
    elif name == "omniglot":
        return SiameseOmniglot(background=train)
    elif name == "celeba":
        return SiameseCelebA(root="data", split="train" if train else "valid")
    else:
        raise ValueError("Unknown dataset name")


# ============ Test Loader ============
if __name__ == "__main__":
    dataset = get_dataset("mnist", train=True)
    print(f"Total samples: {len(dataset)}")
    img1, img2, label = dataset[0]
    print("Image shapes:", img1.shape, img2.shape, "Label:", label.item())
