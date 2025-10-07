import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss:
    L = (1 - Y) * D^2 + Y * {max(0, margin - D)}^2
    where:
        Y = 0 → similar pair
        Y = 1 → dissimilar pair
        D = Euclidean distance between embeddings
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Ensure label is float for multiplication
        label = label.float()

        # Compute Euclidean distance between embeddings
        euclidean_distance = F.pairwise_distance(output1, output2)

        # Compute contrastive loss
        loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss
