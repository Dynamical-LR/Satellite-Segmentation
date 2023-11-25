from torch.utils import data
import cv2
import torch
import numpy
import matplotlib.pyplot as plt

class SegmentationDataset(data.Dataset):

    """
    Implementation of the dataset
    for storing and managing segmentation data

    Parameters:
    -----------

    imgs (list) - list of colored images
    """
    
    def __init__(self, 
        imgs, 
        masks, 
        transformations=None, 
        color_transformations=None
    ):
        self.imgs = imgs
        self.masks = masks
        self.transformations = transformations
        self.color_transformations = color_transformations
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        if idx > len(self.imgs):
            raise IndexError('index is out of range of possible objects')

        img = cv2.imread(self.imgs[idx], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_UNCHANGED)
        
        if img.shape[2] != 3: 
            raise ValueError("image need to have 3 channels, not %s" % str(img.shape[2]))
            
            
        if self.color_transformations is not None:
            content = self.color_transformations(
                image=img
            )
            img = content['image']
                
        if self.transformations is not None:
            content = self.transformations(
                image=img,
                mask=mask,
            )
            img = content['image']
            mask = content['mask']
        
        img = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32)
        mask = torch.from_numpy(mask).to(torch.float32)
        return img, mask
    
    
    
