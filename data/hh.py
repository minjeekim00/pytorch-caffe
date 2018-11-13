import pandas as pd
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image

class HHDataset(Dataset):
    def __init__(self, dir_path, img_path, transform=None):
        #assert tmp_df['img_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all()
        tmp_df = pd.read_csv(dir_path)
        self.img_path = img_path
        self.transform = transform

        self.x = tmp_df['img_name']
        self.y = np.asarray(tmp_df['target'].astype(np.float32))
        
    def __getitem__(self, index):
        img = Image.open(self.img_path.strip() + self.x[index])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(self.y[index], dtype=torch.float32)
        return img, label
    
    def __len__(self):
        return len(self.x.index)