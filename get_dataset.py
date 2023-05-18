import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class VirDataset(Dataset):
    def __init__(self, csv_file,length_match,max_length):

        self.csv_path = os.path.join(csv_file)
        self.df = pd.read_csv(self.csv_path,delimiter="\t")
        self.acc = self.df['accession'].values
        self.feature = self.df['feature'].values
        self.length_match=length_match
        self.max_length=max_length

    def __len__(self):
        return len(self.acc)

    def __getitem__(self, index):
        feature = [int(token) for token in self.feature[index].strip().rsplit(' ')]
        while len(feature)!=self.max_length:
            feature.append(self.length_match)
        return torch.tensor(feature),self.acc[index]

def get_loader(
        annotation_file,
        length_match,
        max_length,
        batch_size=64,
        num_workers=6,
        shuffle=False,
        pin_memory=False,
        drop=False
):

    dataset = VirDataset( annotation_file,length_match,max_length)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop
    )

    return loader

