import cv2
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from pathlib import Path
import os
import shutil


def cut_orig_image(img_path, h, pad=50):
    """Cuts image with big resolution into overlapping squares with size = h. 
    Does reflect padding with pad. Overlap of each square = pad.

    Args:
        img_path (str): path to image
        h (int): Size of square (size of crop)
        pad (int, optional): Number of padding px. Defaults to 50.

    Returns:
        parts : List of cropped squares
        width : Width of padded image
        height : Height of padded image
    """
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    reflect_padding = A.PadIfNeeded(img.shape[0]+2*pad, img.shape[1]+2*pad, border_mode=cv2.BORDER_REFLECT_101)
    img_padded = reflect_padding(image=img)['image']

    height = img_padded.shape[0]
    width = img_padded.shape[1]

    cols = int(np.ceil(width / (h - pad)))
    rows = int(np.ceil(height / (h - pad)))

    parts = []

    for r in range(rows):
        for c in range(cols):
            if c == cols-1:
                x_min = width - h
            else:
                x_min = c * (h-pad)
            
            if r == rows-1:
                y_min = height - h
            else:
                y_min = r * (h-pad)

            crop_transform = A.Crop(x_min, y_min, x_min+h, y_min+h)
            img_crop = crop_transform(image=img_padded)['image']
            parts.append(img_crop)
    return parts, width, height


def concat_result_mask(parts_mask, height, width, h, pad=50):
    """Concats overlaping parts of mask into resulting mask of high resolution. 
    After concat, removes reflect padding

    Args:
        parts_mask : List of masks after inference in model
        height : Height of padded image
        width : Width of padded image
        h (int): Size of square (size of crop)
        pad (int, optional): Number of padding px. Defaults to 50.

    Returns:
        mask_result: Bool mask
    """
    mask_result_padded = np.zeros((height, width))

    cols = int(np.ceil(width / (h - pad)))
    rows = int(np.ceil(height / (h - pad)))

    for i, mask in enumerate(parts_mask):
        c = i % cols
        r = i // cols

        if c == cols-1:
            x_0 = width - h + pad//2
        else:
            x_0 = c * (h-pad) + pad//2
        
        if r == rows-1:
            y_0 = height - h + pad//2
        else:
            y_0 = r * (h-pad) + pad//2

        x_1 = x_0 + h - pad//2
        y_1 = y_0 + h - pad//2
        mask_result_padded[y_0:y_1,x_0:x_1] = mask[pad//2:, pad//2:]
    
    mask_result = mask_result_padded[pad:height-pad, pad:width-pad]
    return mask_result
    
def inference_one_sample(model, img_path, h, pad=50):
    """Inference one high resolution image
    """
    parts, width, height = cut_orig_image(img_path, h, pad)
    
    # model inference, dataloader
    parts_mask = model(parts) # TODO

    mask_result = concat_result_mask(parts_mask, height, width, h, pad)
    return mask_result