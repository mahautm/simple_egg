import numpy as np
import torch
from torch.utils.data import Dataset

# The AttValRecoDataset class is used in the reconstruction game. It takes an input file with a
# space-delimited attribute-value vector per line and  creates a data-frame with the two mandatory
# fields expected in EGG games, namely sender_input and labels.
# In this case, the two fields contain the same information, namely the input attribute-value vectors,
# represented as one-hot in sender_input, and in the original integer-based format in
# labels.
class AttValSumDataset(Dataset):
    def __init__(self, path, n_attributes, n_values):
        frame = np.loadtxt(path, dtype="S10")
        self.frame = []
        for row in frame:
            if n_attributes == 1:
                row = row.split()
            config = list(map(int, row))
            z = torch.zeros((n_attributes, n_values))
            for i in range(n_attributes):
                z[i, config[i]] = 1
            label = torch.tensor(np.sum(list(map(int, row))))  # <-- here be the change
            self.frame.append((z.view(-1), label))

    def get_n_features(self):
        return self.frame[0][0].size(0)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]
