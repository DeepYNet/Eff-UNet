import os
import logging
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DFDCDataset(Dataset):
    def __init__(self, img_dir, labels_dir):
        super().__init__()
        self.img_dir = img_dir
        self.labels_dir = labels_dir

        self.names = []
        self.names_img = []
        self.names_labels = []
        for i in os.listdir(img_dir):
            self.names_img.append(i.split('.')[0])
        for i in os.listdir(labels_dir):
            self.names_labels.append(i.split('.')[0])
        for i in self.names_img:
            if i in self.names_labels:
                self.names.append(i)

        if not self.names:
            raise RuntimeError(f'No input file found in {self.img_dir}, make sure you put your images there')
        
        logging.info(f'Creating dataset with {len(self.names)} examples')

    def __len__(self):
        return len(self.names)

    @classmethod
    def load(self, dir_path, name, is_mask):
        if not is_mask:
            return Image.open(os.path.join(dir_path, name)+'.png')
        else:
            return Image.open(os.path.join(dir_path, name)+'.gif')

    @classmethod
    def preprocess(cls, pil_img, is_mask):
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    def __getitem__(self, index):
        selected = self.names[index]

        img = self.load(self.img_dir, selected, False)
        label = self.load(self.labels_dir, selected, True)

        assert img.size == label.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img,  is_mask=False)
        mask = self.preprocess(label, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }