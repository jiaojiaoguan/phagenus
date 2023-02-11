import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class VirDataset(Dataset):
    def __init__(self, root_dir, csv_file,buqi,max_length):
        self.root_dir = root_dir
        self.csv_path = os.path.join(root_dir, csv_file)
        self.df = pd.read_csv(self.csv_path,delimiter="\t")
        self.acc = self.df['accession'].values
        self.labels = self.df['genusid'].values
        self.feature = self.df['feature'].values
        self.buqi=buqi
        self.max_length=max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        feature = [int(token) for token in self.feature[index].strip().rsplit(' ')]
        while len(feature)!=self.max_length:
            feature.append(self.buqi)
        return torch.tensor(feature), torch.tensor(self.labels[index]), self.acc[index]

def get_loader(
        root_folder,
        annotation_file,
        buqi,
        max_length,
        batch_size=64,
        num_workers=6,
        shuffle=False,
        pin_memory=False,
        drop=False
):

    dataset = VirDataset(root_folder, annotation_file,buqi,max_length)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop
    )

    return loader

