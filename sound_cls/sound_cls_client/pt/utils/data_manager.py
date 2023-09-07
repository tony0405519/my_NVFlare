import os
import pandas as pd
import numpy as np

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate

from pt.utils.dataset import FolderDataset
import soundfile as sf
import glob as glob

class CSVDataManager(object):

    def __init__(self, datapath, train=False):
        
        # "/home/NVFlare/sound_cls/sound_datasets/urbansound8k/"
        self.dir_path = datapath
        self.files, self.labels = [], [] 
        self.files_to_remove = []
        self.load_datas(min_sec=1, train=train) 

    def load_datas(self, min_sec=0.5, train=False):
        self.files, self.labels = [], []
        for file in os.listdir(self.dir_path):
            if file.endswith(".wav"):
                if train:
                    self.files_to_remove.append(os.path.join(self.dir_path, file))
                f = sf.SoundFile(os.path.join(self.dir_path, file))
                if f.frames/f.samplerate >= min_sec:
                    self.files.append(os.path.join(self.dir_path, file))
                    self.labels.append(int(file.split('-')[1]))

    def get_loader(self, transfs, batch_size=64, shuffle=True, num_workers=4):
        dataset = FolderDataset(self.files, self.labels, transforms=transfs)
        loader = data.DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, drop_last=True, collate_fn=self.pad_seq)
        return loader

    def remove_files(self):
        for f in self.files_to_remove:
            os.remove(f)
        self.files_to_remove = []

    def pad_seq(self, batch):
        # sort_ind should point to length
        sort_ind = 0
        sorted_batch = sorted(batch, key=lambda x: x[0].size(sort_ind), reverse=True)
        seqs, srs, labels = zip(*sorted_batch)
        
        #lengths, srs, labels = map(torch.LongTensor, [[x.size(sort_ind) for x in seqs], srs, labels])
        # lengths = map(torch.LongTensor, [x.size(sort_ind) for x in seqs])
        lengths = torch.LongTensor([x.size(sort_ind) for x in seqs])
        # srs = map(torch.LongTensor, srs)
        srs = torch.LongTensor(srs)
        # labels = map(torch.LongTensor, labels)
        labels = torch.LongTensor(labels)

        # seqs_pad -> (batch, time, channel) 
        seqs_pad = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
        #seqs_pad = seqs_pad_t.transpose(0,1)
        return seqs_pad, lengths, srs, labels
