import os
import pandas as pd
import gdown
import PIL
import tarfile
import torch
from torchvision import transforms

class CelebA(torch.utils.data.Dataset):
    def __init__(self, root, split='train', target='Blond_Hair', bias_attr='Male', unbiased=True, seed=42, train_proc=False):
        path = root
        if not os.path.isdir(os.path.join(path, 'celeba')):
            self.download_dataset(path)
        path = os.path.join(path, 'celeba')
        self.split = split
        self.train_proc=train_proc

        split_df = pd.read_csv(os.path.join(path, 'list_eval_partition.csv'))
        splits = {
            'train': 0,
            'valid': 1,
            'test': 2
        }
        partition_idx = split_df['split'] == splits[split]

        self.attr_df = pd.read_csv(os.path.join(path, 'list_attr_celeba.csv'), sep=' ').replace(-1, 0)
        
        # keep only relevant split train/val/test
        self.attr_df = self.attr_df[partition_idx]

        #swap male/female
        self.attr_df['Male'] = ~self.attr_df['Male']+2

        if split in ['valid', 'test']:
            min_size = self.attr_df.groupby([target, bias_attr]).count().min()['image']

            # construct unbiased dataset (equal size for (at, ab))
            unbiased_df = self.attr_df.groupby([target, bias_attr]).apply(lambda group: group.sample(min_size, random_state=seed)).reset_index(drop=True)

            # remove bias-confictling pairs
            bias_conflicting_df = unbiased_df[unbiased_df[target] != unbiased_df[bias_attr]]

            if unbiased:
                self.attr_df = unbiased_df
            else:
                self.attr_df = bias_conflicting_df

        #print(self.attr_df.groupby([target, bias_attr]).count())


        self.target = target
        self.bias_attr = bias_attr
        self.path = path

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        T_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        T_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


        self.T = T_train if (split == 'train' or self.train_proc) else T_test

    def download_dataset(self, path):
        url = "https://drive.google.com/uc?id=1ebDzE4vsjPB4klNyTywjrZqGhUsFxZqb"
        output = os.path.join(path, 'celeba.tar.gz')
        print(f'=> Downloading CelebA dataset from {url}')
        gdown.download(url, output, quiet=False)

        print('=> Extracting dataset..')
        tar = tarfile.open(os.path.join(path, 'celeba.tar.gz'), 'r:gz')
        tar.extractall(path=path)
        tar.close()
        os.remove(output)

    def __getitem__(self, index):
        data = self.attr_df.iloc[index]
        img_name = data['image']
        bias = data[self.bias_attr]
        target_attr = data[self.target]

        image = PIL.Image.open(os.path.join(self.path, 'img_align_celeba', img_name))
        return self.T(image), (target_attr, bias)

    def __len__(self):
        return len(self.attr_df)
