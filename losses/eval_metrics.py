from torch import nn
import numpy

class DiceLoss(nn.Module):
    """
    Implementation of Dice Loss Function
    """
    def forward(self, predicted_mask, mask: numpy.ndarray):
        intersection = sum((predicted_mask & mask).flatten())
        union = sum((predicted_mask + mask).flatten())
        return 1 - (2 * intersection / union)


class F1Score(nn.Module):
    """
    Implementation of the F1 Score metric
    """
    def _calculate_recall(self, pred_mask, mask):
        pass 

    def _calculate_precision(self, pred_mask, mask):
        pass 

    def forward(self, predicted_mask, mask):
        recall = self._calculate_recall(predicted_mask, mask)
        precision = self._calculate_precision(predicted_mask, mask)
        return (2 * precision * recall) / precision + recall