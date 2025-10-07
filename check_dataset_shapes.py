import torch
from siamese_datasets import get_dataset

# List of all datasets to test
datasets_list = ["mnist", "fmnist", "omniglot", "celeba"]

for dataset_name in datasets_list:
    print(f"\n=== Checking dataset: {dataset_name.upper()} ===")
    dataset = get_dataset(dataset_name, train=True)

    # Fetch 3 random samples
    for i in range(3):
        img1, img2, label = dataset[i]

        # Print shape info
        print(f" Sample {i+1}:")
        print(f"   img1 shape: {tuple(img1.shape)} | img2 shape: {tuple(img2.shape)} | Label: {label.item()}")

        # Channel check
        if img1.shape[0] == 3:
            print("   ✅ RGB verified (3 channels)")
        else:
            print("   ❌ Not RGB! Channels =", img1.shape[0])

        # Size check
        if img1.shape[1:] == (128, 128):
            print("   ✅ Size verified (128x128)")
        else:
            print("   ❌ Wrong size! Found =", img1.shape[1:])

print("\n✅ Dataset verification complete.")
