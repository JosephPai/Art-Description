import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class CaptionDatasetPara(Dataset):
    def __init__(self, data_folder, data_name, split, transform=None):
        self.split = split
        assert self.split in {'TRAIN', 'TRAINVAL', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.imgs)

    def __getitem__(self, i):
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.stack([torch.LongTensor(self.captions[0][i]),
                               torch.LongTensor(self.captions[1][i]),
                               torch.LongTensor(self.captions[2][i])], dim=0)

        caplen = torch.stack([torch.LongTensor([self.caplens[0][i]]),
                              torch.LongTensor([self.caplens[1][i]]),
                              torch.LongTensor([self.caplens[2][i]])], dim=0)

        all_captions = torch.stack(([torch.LongTensor(self.captions[0][((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)]),
                                     torch.LongTensor(self.captions[1][((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)]),
                                     torch.LongTensor(self.captions[2][((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])]))
        return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
