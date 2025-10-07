import matplotlib.pyplot as plt
import torch
from siamese_datasets import get_dataset


def show_pair(img1, img2, label, dataset_name):
    """Visualize a pair of images with their similarity label."""
    plt.figure(figsize=(4, 2))

    # Show first image
    plt.subplot(1, 2, 1)
    plt.imshow(img1.permute(1, 2, 0))
    plt.axis('off')
    plt.title("Image 1")

    # Show second image
    plt.subplot(1, 2, 2)
    plt.imshow(img2.permute(1, 2, 0))
    plt.axis('off')
    plt.title(f"Image 2\nLabel: {'Same' if label.item() == 1 else 'Different'}")

    plt.suptitle(f"Dataset: {dataset_name.upper()}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # List only available datasets (since CASIA/VGGFace2 are not integrated yet)
    datasets_list = ["mnist", "fmnist", "omniglot", "celeba"]

    for dataset_name in datasets_list:
        print(f"\n=== Showing pairs from {dataset_name.upper()} ===")

        try:
            dataset = get_dataset(dataset_name, train=True)
        except Exception as e:
            print(f"❌ Failed to load {dataset_name}: {e}")
            continue

        # Show 2 sample pairs for each dataset
        for i in range(2):
            img1, img2, label = dataset[i]

            # Ensure tensors are in CPU for matplotlib
            if isinstance(img1, torch.Tensor):
                img1 = img1.cpu()
                img2 = img2.cpu()

            show_pair(img1, img2, label, dataset_name)
