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
        
    def forward(y_pred: torch.Tensor, y_true: torch.Tensor, threshold=0.5):
        
        y_pred_bin = (y_pred > threshold).float()
        y_true_bin = y_true.float()
      
        # Calculate true positives, false positives, and false negatives
        true_positives = torch.sum((y_pred_bin == 1) & (y_true_bin == 1)).item()
        false_positives = torch.sum((y_pred_bin == 1) & (y_true_bin == 0)).item()
        false_negatives = torch.sum((y_pred_bin == 0) & (y_true_bin == 1)).item()

        # Calculate precision, recall, and F1 score
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        return f1

