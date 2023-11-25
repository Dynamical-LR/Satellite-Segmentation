from torch import nn 
import numpy
from torch.nn import functional as F
import torch


class DiceLoss(nn.Module):
    
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth # smoothing factor for avoiding division by zero

    def forward(self, y_pred, y_true):

        # Flatten the tensors
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)

        # Calculate intersection and union
        intersection = torch.sum(y_pred_flat * y_true_flat)
        union = torch.sum(y_pred_flat) + torch.sum(y_true_flat)

        # Calculate Dice coefficient
        dice_coefficient = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Calculate Dice Loss
        dice_loss = 1 - dice_coefficient

        return dice_loss

class FocalLoss(nn.Module):

    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # Flatten the predictions and ground truth masks
        
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)

        # Calculate binary cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(y_pred_flat, y_true_flat, reduction='none')

        # Calculate focal loss
        focal_loss = (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss

        # Calculate mean over all elements
        mean_loss = torch.mean(focal_loss)

        return mean_loss

class ComboLoss(nn.Module):
    """
    Weighted combination of dice loss and focal loss

    Parameters:
    ----------
    dice_prop - proportion percentage for focal loss
    focal_prop - proportion percentage for dice loss
    focal_gamma - gamma value for focal loss
    dice_eps - dice coefficinet epsilon for preventing division by zero
    """
    def __init__(self, dice_prop: float, focal_prop: float, focal_gamma: float):
        super(ComboLoss, self).__init__()

        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.dice_loss = DiceLoss()
        self.dice_prop = dice_prop
        self.focal_prop = focal_prop

        if dice_prop + focal_prop != 1.0:
            raise ValueError(
            msg='first and second props must equal to 1 in summary'
            )
    
    def forward(self, predicted_mask_of_probs: torch.Tensor, true_mask: torch.Tensor):
        focal_loss = self.focal_loss(predicted_mask_of_probs, true_mask)
        dice_loss = self.dice_loss(predicted_mask_of_probs, true_mask)
        return self.focal_prop * focal_loss + self.dice_prop * dice_loss



    
    