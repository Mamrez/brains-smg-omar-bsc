import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple

class RNNPreparedDataset(Dataset):
    def __init__(self, filename: str, sequence_length: int, steps: int = 1) -> None:
        """
        Initialization of the dataset. It loads a posprocessed_data.npz file into memory.
        The targets of this file are divided by the amplification correction factor, so that
        data is made setup independent.

        Parameters
        ----------
        filename : str
            Folder and filename where the posprocessed_data.npz is.
        sequence_length : int
            The length of sequences for the RNN.
        steps : int
            It allows to skip parts of the data when loading it into memory. The number indicates
            how many items will be skipped in between. By default, step number is one (no values
            are skipped). E.g., if steps = 2, and the inputs are [0, 1, 2, 3, 4, 5, 6]. The only
            inputs taken into account would be: [0, 2, 4, 6].
        """
        self.sequence_length = sequence_length
        self.inputs, targets, self.sampling_configs = self.load_data_from_npz(filename, steps)
        self.targets = targets / self.sampling_configs["driver"]["amplification"]
        self.inputs = self.format(self.inputs)
        self.targets = self.format(self.targets)

        assert len(self.inputs) == len(self.targets), "Inputs and Outputs have NOT the same length"

    def __len__(self) -> int:
        """
        Overwrites the __len__ method from the super class torch.utils.data.

        Returns
        -------
        int
            Size of the whole dataset.
        """
        return len(self.inputs) - self.sequence_length + 1

    def __getitem__(self, index: int) -> Tuple[np.array, np.array]:
        """
        Overwrites the __getitem__ method from the super class torch.utils.data.
        The method supports fetching a data sample for a given key.

        Parameters
        ----------
        index : int
            Index corresponding to the place of the data in the dataset.

        Returns
        -------
        tuple
            Inputs and targets of the dataset corresponding to the given index.
        """
        start_idx = index
        end_idx = start_idx + self.sequence_length

        input_seq = self.inputs[start_idx:end_idx]
        target_value = self.targets[end_idx - 1]

        return input_seq, target_value

    def load_data_from_npz(self, filename: str, steps: int) -> Tuple[np.array, np.array, dict]:
        """
        Loads the inputs, targets, and sampling configurations from a given postprocessed_data.npz file.

        Parameters
        ----------
        filename : str
            Folder and filename where the posprocessed_data.npz is.
        steps : int
            It allows to skip parts of the data when loading it into memory. The number indicates
            how many items will be skipped in between. By default, step number is one (no values
            are skipped). E.g., if steps = 2, and the inputs are [0, 1, 2, 3, 4, 5, 6]. The only
            inputs taken into account would be: [0, 2, 4, 6].

        Returns
        -------
        inputs : np.array
            Input waves sent to the activation electrodes of the device during sampling.
        outputs : np.array
            Raw output data from the readout electrodes of the device during sampling,
            corresponding to the input.
        sampling_configs : dict
            Dictionary containing the sampling configurations with which the data was acquired.
        """
        print("\n* Loading data from file:\n" + filename)
        with np.load(filename, allow_pickle=True) as data:
            sampling_configs = dict(data["sampling_configs"].tolist())
            inputs = data["inputs"][::steps]
            outputs = data["outputs"][::steps]
            print(f"\t- Shape of inputs:  {inputs.shape}\n\t- Shape of outputs: {outputs.shape}\n")
            print(f"* Sampling configs has the following keys:\n\t{sampling_configs.keys()}\n")
        return inputs, outputs, sampling_configs

    @staticmethod
    def format(data):
        return torch.tensor(data, dtype=torch.float32)


def collate_rnn_batches(sequence_length):
    def collate_fn(batch):
        inputs, targets = zip(*batch)
        data = np.vstack([np.hstack((x, y[:, None])) for x, y in zip(inputs, targets)])
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
        input_seq, target_value = data[start_idx:end_idx, :-1], data[end_idx - 1, -1]
        input_sequences.append(input_seq)
        target_values.append(target_value)
    return np.array(input_sequences), np.array(target_values)
