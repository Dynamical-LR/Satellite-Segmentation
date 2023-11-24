from torch import nn
import numpy

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
    """
    def _calculate_recall(self, pred_mask, mask):
        tp = len(pred_mask[pred_mask == mask == 1].flatten())
        fn = len(pred_mask[pred_mask != mask == 1].flatten())  # Fix: Count false negatives correctly
        return tp / (fn + tp)

    def _calculate_precision(self, pred_mask, mask):
        tp = len(pred_mask[pred_mask == mask == 1].flatten())
        fp = len(pred_mask[pred_mask != mask == 0].flatten())  # Fix: Count false positives correctly
        return tp / (fp + tp)

    def forward(self, predicted_mask, mask):
        recall = self._calculate_recall(predicted_mask, mask)
        precision = self._calculate_precision(predicted_mask, mask)
        return (2 * precision * recall) / (precision + recall)
