import os
import re
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

import utils


class MyDataset(Dataset):
    def __init__(self, file_list: list, name2label, transform_flag=True):
        self.file_list = file_list
        self.name2label = name2label
        self.transform_flag = transform_flag
        self.aug_transforms = T.Compose([
            # T.Resize((256, 256)),
            T.ColorJitter(),
            T.RandomRotation(30),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])

    def __getitem__(self, item):
        image, label = self.transform(self.file_list[item], transform_flag=self.transform_flag)
        return image, label

    # @classmethod
    def transform(self, filename, transform_flag):
        # preprocess (default Identity)
        image = Image.open(filename)
        if transform_flag:
            image = self.aug_transforms(image)
        image = T.ToTensor()(image)
        class_name = re.findall('\w+_[0-9]', os.path.split(filename)[-1])[0][:-2]
        label = self.name2label[class_name]
        return image, label

    def __len__(self):
        return len(self.file_list)
