import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingNet(nn.Module):
    """
    Flexible embedding network that works for any RGB image size.
    Converts image into a 256-D embedding.
    """
    def __init__(self):
        super(EmbeddingNet, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),   # (3,H,W) → (32,H-4,W-4)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                # downsample ×2

            nn.Conv2d(32, 64, kernel_size=5),  # (64, (H/2)-4, (W/2)-4)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                # downsample ×2

            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        # We’ll determine flatten size dynamically later
        self.fc1 = nn.Linear(1, 1)  # placeholder (will reset below)
        self.fc2 = nn.Linear(512, 256)

    def forward_once(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Initialize fc1 dynamically on first forward
        if self.fc1.in_features == 1:
            self.fc1 = nn.Linear(x.size(1), 512).to(x.device)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, x):
        return self.forward_once(x)


class SiameseNetwork(nn.Module):
    """
    Siamese Network using shared EmbeddingNet.
    """
    def __init__(self, embedding_net):
        super(SiameseNetwork, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2
