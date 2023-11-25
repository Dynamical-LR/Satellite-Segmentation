from torch import nn
import numpy
import torch

class DiceCoefficient(nn.Module):
    """
    Implementation of Dice Loss Function
    """
    def forward(self, predicted_mask, mask: numpy.ndarray):
        intersection = sum((predicted_mask & mask).flatten())
        union = sum((predicted_mask + mask).flatten())
        return (2 * intersection / union)

class F1Score(nn.Module):
    
    """
    Implementation of the Averaged F1 Score metric
    Compute the F1 score between predicted and true binary masks.

    Parameters:
        y_pred (torch.Tensor): Predicted binary mask (0 or 1).
        y_true (torch.Tensor): Ground truth binary mask (0 or 1).
        threshold (float): Threshold for binarizing the predicted mask.

    Returns:
        float: F1 score.
    """
    def __init__(self):
        super(F1Score, self).__init__()
        self.epsilon = 0.00001
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, threshold=0.5):
        
        y_true = y_true.numpy().flatten()
        y_pred = y_pred.numpy().flatten()
        
        # Calculate true positives, false positives, and false negatives
        tp = (y_true * y_pred).sum(axis=0)
        fp = ((1 - y_true) * y_pred).sum(axis=0)
        fn = (y_true * (1 - y_pred)).sum(axis=0)

        # Calculate precision, recall, and F1 score
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)

        # Average over classes (assuming the input tensors are batched)
        f1 = f1.mean()
        return max(0, f1.item())
