from torch import nn 
import numpy
from torch import functional as F
import torch


class DiceLoss(nn.Module):
    """
    Implementation of the Dice Loss Function
    
    Parameters:
    -----------
    
    eps - constant for preventing division by zero
    """
    def __init__(self, eps: float):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, predicted_mask_of_probs, true_mask):
        predicted_mask = numpy.argmax(predicted_mask_of_probs, axis=1)
        intersection = torch.sum(predicted_mask * true_mask)
        union = torch.sum(predicted_mask) + torch.sum(true_mask)
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        loss = 1.0 - dice
        return loss


class FocalLoss(nn.Module):

    def __init__(self, gamma):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, predicted_mask_of_probs, true_mask):
        
        # Binary cross entropy loss term
        bce_loss = F.binary_cross_entropy(predicted_mask_of_probs, true_mask, reduction='none')
        predicted_binary_mask = torch.argmax(predicted_mask_of_probs, keepdim=True, dim=1)

        # Focal loss term
        focal_loss = (1 - predicted_binary_mask) ** self.gamma * bce_loss

        # Mask to only consider positive samples (where actual_mask == 1)
        positive_mask = (true_mask == 1)

        # Combine the losses for positive and negative samples
        loss = torch.sum(positive_mask * focal_loss + (1 - positive_mask) * bce_loss)

        return loss

class ComboLoss(nn.Module):
    """
    Weighted combination of dice loss and focal loss

    Parameters:
    ----------
    first_prop - proportion percentage for focal loss
    second_prop - proportion percentage for dice loss
    """
    def __init__(self, first_prop: float, second_prop: float, focal_gamma: float):
        super(ComboLoss, self).__init__()

        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.dice_loss = DiceLoss()
        self.first_prop = first_prop
        self.second_prop = second_prop

        if first_prop + second_prop != 1.0:
            raise ValueError(
            msg='first and second props must equal to 1 in summary'
            )
    
    def forward(self, predicted_mask_of_probs: numpy.ndarray, true_mask: numpy.ndarray):
        focal_loss = self.focal_loss(predicted_mask_of_probs, true_mask)
        dice_loss = self.dice_loss(predicted_mask_of_probs, true_mask)
        return self.first_prop * focal_loss + self.second_prop * dice_loss


