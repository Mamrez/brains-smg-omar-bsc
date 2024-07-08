"""
File containing a class for loading sampling data as a RNN dataset.
"""
import torch
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from brainspy.utils.pytorch import TorchUtils
from typing import Tuple, List
import matplotlib.pyplot as plt
from torch.utils.data import Subset

class RNNPreparedDataset(Dataset):
    def __init__(self, dataset, sequence_length):
        self.dataset = dataset
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample, target = self.dataset[idx]
        sample = sample.cpu().numpy() if sample.is_cuda else sample.numpy()
        target = target.cpu().numpy() if target.is_cuda else target.numpy()
        data = np.hstack((sample, target))
        return data

def collate_rnn_batches(sequence_length):
    def collate_fn(batch):
        data = np.vstack(batch)
        X, y = prepare_rnn_sequences(data, sequence_length)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return X_tensor, y_tensor
    return collate_fn


def prepare_rnn_sequences(data, sequence_length):
    input_sequences, target_values = [], []
    for start_idx in range(len(data)):
        end_idx = start_idx + sequence_length
        if end_idx > len(data):
            break
        input_seq, target_value = data[start_idx:end_idx, :-1], data[end_idx-1, -1]
        input_sequences.append(input_seq)
        target_values.append(target_value)
    return np.array(input_sequences), np.array(target_values)