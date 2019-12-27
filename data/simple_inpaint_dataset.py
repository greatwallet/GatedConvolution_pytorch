import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms

from .inpaint_dataset import InpaintDataset

class SimpleInpaintDataset(InpaintDataset):
    r""" A simple implementation of inpaint dataset 
    which only supports pre-defined masks
    
    """
    
    def __init__(self, img_flist_path, mask_flist_path,
                resize_shape=(256, 256), transforms_oprs=['to_tensor']):

        with open(img_flist_path, 'r') as f:
            self.img_paths = f.read().splitlines()
        
        with open(mask_flist_path, 'r') as f:
            self.mask_paths = f.read().splitlines()
        
        # assure that the mask and the image path list are of same length
        assert(len(self.mask_paths) == len(self.img_paths))
        
        self.resize_shape = resize_shape
        self.transform_initialize(resize_shape, transforms_oprs)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        # create the paths for images and masks
        img_path = self.img_paths[index]
        img = self.transforms_fun(self.read_img(img_path))
        mask_path = self.mask_paths[index]
        mask = self.read_mask(mask_path)
        mask = self.transforms_fun(mask)

        return img * 255, mask
    
    def read_mask(self, path):
        """
        Read mask
        """
        mask = Image.open(path).convert("1")
        return mask