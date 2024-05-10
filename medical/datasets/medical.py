from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm
from .builder import DATASETS
from .base_dataset import BaseDataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from query_strategies.utils import UnNormalize
from sklearn.preprocessing import LabelEncoder

class MBH(Dataset):
    def __init__(self, train: bool = True, transform=None, target_transform=None):
        self.train = train
        
        self.img_info = pd.read_csv('/kaggle/input/mbh-al/AL/data.csv')
        self.img_dir = '/kaggle/input/mbh-al/AL'
        
        lbe = LabelEncoder()
        lbe.fit(self.img_info['labels'])
        self.classes = lbe.transform(lbe.classes_)
        
        self.targets = lbe.transform(self.img_info['labels'])
    
        self.data = []
        img_names = self.img_info['images'].tolist()
        for img_name in tqdm(img_names):
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path)
            image = np.array(image)
            self.data.append(image)
            
        self.data = np.array(self.data)
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            x = transform(x)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

@DATASETS.register_module()
class medical(BaseDataset):
    def __init__(self,
                 data_path=None,
                 initial_size=None, seed=0):
        super(medical, self).__init__(data_path, initial_size, seed)

    def load_data(self):
        raw = MBH(train=True)

        n = len(raw.targets)
        np.random.seed(self.seed)
        p = np.random.permutation(n)

        num_tr = int(0.8*n)
        num_vl = int(0.1*n)
        num_te = n - num_tr - num_vl
        
        val_idx_list = p[num_tr: num_tr + num_vl]
        test_idx_list = p[num_tr + num_vl:]


        self.DATA_INFOS['train_full'] = [{'no': i, 'img': raw.data[i],
                                          'gt_label': int(raw.targets[i])} for i in range(n)]
        
        self.DATA_INFOS['val'] = np.array(self.DATA_INFOS['train_full'])[val_idx_list].tolist()
        self.DATA_INFOS['test'] = np.array(self.DATA_INFOS['train_full'])[test_idx_list].tolist()
        
        self.DATA_INFOS['train_full'] = np.delete(np.array(self.DATA_INFOS['train_full']), 
                                                  np.concatenate((val_idx_list, test_idx_list), axis=0)).tolist()
        
        self.num_samples = num_tr + num_vl + num_te
        self.INDEX_LB = np.zeros(num_tr, dtype=bool)
        self.CLASSES = raw.classes

    def prepare_data(self, idx, split, transform=None, aug_transform=None):
        x, y = self.DATA_INFOS[split][idx]['img'], self.DATA_INFOS[split][idx]['gt_label']
        x = Image.fromarray(x)
        
        if aug_transform is not None:
            x = aug_transform(x)
        if transform is None:
            x = self.TRANSFORM[split](x)
        else:
            x = np.array(x)
            augmented = transform(image=x)
            x = augmented['image']
        return x, y, self.DATA_INFOS[split][idx]['no'], idx

    @property
    def default_train_transform(self):
        return A.Compose([
           A.VerticalFlip(p=.5),
           A.HorizontalFlip(p=.5),
           A.HueSaturationValue(hue_shift_limit=(-25,0), sat_shift_limit=0, val_shift_limit=0,p=1),
           A.Rotate(p=1, border_mode=cv2.BORDER_CONSTANT,value=0),
           A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
           ToTensorV2()
        ])
    @property
    def default_val_transform(self):
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    @property
    def inverse_transform(self):
        return A.Compose([
            UnNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            A.ToPILImage()
        ])

    def get_raw_data(self, idx, split='train'):
        transform = self.default_val_transform
        x = self.DATA_INFOS[split][idx]['img']
        x = transform(x)
        return x