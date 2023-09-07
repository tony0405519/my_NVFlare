import os
import pandas as pd
import numpy as np
import soundfile as sf
import torch.utils.data as data

class FolderDataset(data.Dataset):
    def __init__(self, files, labels, transforms=None):
        
        self.files = files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        
        data, sr = sf.read(self.files[index])
        label = self.labels[index]

        if self.transforms is not None:
            audio = self.transforms(data)
            return audio, sr, label

        return data, label
