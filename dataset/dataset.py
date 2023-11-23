from torch.utils import data

class SegmentationDataset(data.Dataset):

    """
    Implementation of the dataset object
    for storing and managing segmentation data

    Parameters:
    -----------

    imgs (list) - list of grayscaled images
    """
    def __init__(self, imgs, masks, transformations=None):
        self.imgs = imgs
        self.masks = masks
        self.transformations = transformations
        
    def __len__(self):
        return len(self.imgs)

    def augment(self):
        """
        Functions augments images using provided
        'transformations'

        NOTE:
            after applying the transformations
            both image and mask are going to be modified
        """
        for idx in range(len(self.imgs)):
            content = self.transformations(
                image=self.imgs[idx], 
                mask=self.masks[idx]
            )
            self.imgs[idx] = content['image']
            self.masks[idx] = content['mask']

    def __getitem__(self, idx):

        if idx > len(self.imgs):
            raise IndexError('index is out of range of possible objects')

        img = self.imgs[idx]
        mask = self.masks[idx]
        
        return img, mask