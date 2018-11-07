import pandas as pd
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image

class ENTDataset(Dataset):
    def __init__(self, dir_path, img_path, transform=None):
        #assert tmp_df['img_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all()
        tmp_df = pd.read_csv(dir_path)
        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.transform = transform

        self.x = tmp_df['img_name']
        self.y = self.mlb.fit_transform(tmp_df['direction'].str.split()).astype(np.float32)
    
    def __getitem__(self, index):
        img = Image.open(self.img_path.strip() + self.x[index])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        label = torch.from_numpy(self.y[index])
        return img, label
    
    def __len__(self):
        return len(self.x.index)