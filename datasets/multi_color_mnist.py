# File:                     config.py
# Code from:                https://github.com/zhihengli-UR/DebiAN/blob/main/datasets/multi_color_mnist.py
# Last revised by:          Rémi Nahon
#            date:          2024/7/18
# ==========================================================================


import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class MultiColorMNIST(Dataset):
    attribute_names = ['digit', 'LColor', 'RColor']
    basename = "multi_color_mnist"
    target_attr_index = 0
    left_color_bias_attr_index = 1
    right_color_bias_attr_index = 2

    def __init__(
        self,
        root,
        split,
        left_color_skew,
        right_color_skew,
        severity,
        transform=ToTensor(),
    ):
        super().__init__()

        assert split in ['train', 'test']
        if split == "test":
            split = "valid"
        assert left_color_skew in [0.005, 0.01, 0.02, 0.05, 0.9]
        assert right_color_skew in [0.005, 0.01, 0.02, 0.05, 0.9]
        assert severity in [1, 2, 3, 4]

        root = os.path.join(
            root,
            self.basename,
            f"ColoredMNIST-SkewedA{left_color_skew}-SkewedB{right_color_skew}-Severity{severity}",
        )
        assert os.path.exists(root), f"{root} does not exist"

        data_path = os.path.join(root, split, "images.npy")
        self.data = np.load(data_path)

        attr_path = os.path.join(root, split, "attrs.npy")
        self.attr = torch.LongTensor(np.load(attr_path))

        self.transform = transform

    def __len__(self):
        return self.attr.size(0)

    def __getitem__(self, idx):
        image, attr = self.data[idx], self.attr[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, (attr[0], attr[1], attr[2])


import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_class=10, feature_pos="post"):
        super(MLP, self).__init__()
        self.extracter = nn.Sequential(
            nn.Linear(3 * 28 * 28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
        )
        self.fc = nn.Linear(100, num_class)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if feature_pos not in ['pre', 'post', 'logits']:
            raise ValueError(feature_pos)

        self.feature_pos = feature_pos

    def forward(self, x, logits_only=True):
        x = x.view(x.size(0), -1) / 255
        # x = torch.flatten(x, 1)/255 
        pre_gap_feats = self.extracter(x)
        post_gap_feats = pre_gap_feats
        logits = self.fc(post_gap_feats)

        if logits_only:
            return logits

        elif self.feature_pos == "pre":
            feats = pre_gap_feats
        elif self.feature_pos == "post":
            feats = post_gap_feats
        else:
            feats = logits
        return logits, feats


class MLP2(nn.Module):
    def __init__(self, num_class=10, return_feat=False):
        super(MLP2, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(3 * 28 * 28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(100, num_class)
        self.return_feat = return_feat

    def forward(self, x):
        x = x.view(x.size(0), -1) / 255
        feat = x = self.feature(x)
        x = self.classifier(x)

        if self.return_feat:
            return x, feat
        else:
            return x
