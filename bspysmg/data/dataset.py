import time
import psutil
import torch
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from brainspy.utils.pytorch import TorchUtils
from typing import Tuple, List
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import os
from sklearn.preprocessing import MinMaxScaler


os.environ["CUDA_VISIBLE_DEVICES"]="0"
class ModelDataset(Dataset):
    def __init__(self, filename: str, steps: int = 10, max_datapoints: int = None) -> None:
        self.inputs, targets, self.sampling_configs = self.load_data_from_npz(filename, steps)
        self.targets = targets / self.sampling_configs["driver"]["amplification"]
        
        print(f"Number of zeros in raw targets: {np.sum(self.targets == 0)}")
        print(f"Number of zeros in raw inputs: {np.sum(self.inputs == 0)}")

        plot_targets(self.targets, title="Target Values over Samples not normal", xlabel="Sample Index", ylabel="Target Value", filename="1.png")

        # Normalize inputs and targets using MinMaxScaler   
        self.inputs = self.normalize(self.inputs)
        self.targets = self.normalize(self.targets)
        
        print(f"Number of zeros in normalized targets: {np.sum(self.targets == 0)}")
        print(f"Number of zeros in normalized inputs: {np.sum(self.inputs == 0)}")

        plot_targets(self.targets, title="Target Values over Samples normal", xlabel="Sample Index", ylabel="Target Value", filename="2.png")

        self.inputs = TorchUtils.format(self.inputs)
        self.targets = TorchUtils.format(self.targets)
        
        print(f"Number of zeros in formatted targets: {torch.sum(self.targets == 0).item()}")
        print(f"Number of zeros in formatted inputs: {torch.sum(self.inputs == 0).item()}")

        assert len(self.inputs) == len(self.targets), "Inputs and Outputs have NOT the same length"

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> Tuple[np.array]:
        return (self.inputs[index, :], self.targets[index, :])

    def load_data_from_npz(self, filename: str, steps: int) -> Tuple[np.array, np.array, dict]:
        print("\n* Loading data from file:\n" + filename)
        with np.load(filename, allow_pickle=True) as data:
            sampling_configs = dict(data["sampling_configs"].tolist())
            inputs = data["inputs"][::steps]
            outputs = data["outputs"][::steps]
            print(f"\t- Shape of inputs:  {inputs.shape}\n\t- Shape of outputs: {outputs.shape}\n")
            print(f"* Sampling configs has the following keys:\n\t{sampling_configs.keys()}\n")
        return inputs, outputs, sampling_configs

    def normalize(self, data: np.array) -> np.array:
        scaler = MinMaxScaler()
        return scaler.fit_transform(data)

def get_info_dict(training_configs: dict, sampling_configs: dict) -> dict:
    info_dict = {}
    info_dict["model_structure"] = training_configs["model_structure"].copy()
    info_dict["electrode_info"] = sampling_configs["electrode_info"].copy()
    del training_configs["model_structure"]
    info_dict["training_configs"] = training_configs.copy()
    del sampling_configs["electrode_info"]
    info_dict["sampling_configs"] = sampling_configs.copy()
    return info_dict

def get_dataloaders(configs: dict) -> Tuple[List[torch.utils.data.DataLoader], float, dict]:
    assert 'data' in configs and 'dataset_paths' in configs['data']
    assert isinstance(configs['data']['dataset_paths'], list), "Paths for datasets should be passed as a list"
    assert configs['data']['dataset_paths'] != [], "Empty paths for datasets"
    datasets = []
    info_dict = None
    amplification = None
    dataset_names = ['train', 'validation', 'test']

    if len(configs['data']['dataset_paths']) > 1:
        for i in range(len(configs['data']['dataset_paths'])):
            if configs['data']['dataset_paths'][i] is not None:
                dataset = ModelDataset(configs['data']['dataset_paths'][i], steps=configs['data']['steps'])

                if i > 0:
                    amplification_aux = TorchUtils.format(info_dict["sampling_configs"]["driver"]["amplification"])
                    assert torch.eq(amplification_aux, amplification).all(), (
                        "Amplification correction factor should be the same for all datasets. Check if all datasets come from the same setup.")
                    info_dict[dataset_names[i] + '_sampling_configs'] = dataset.sampling_configs
                else:
                    info_dict = get_info_dict(configs, dataset.sampling_configs)
                amplification = TorchUtils.format(info_dict["sampling_configs"]["driver"]["amplification"])
                datasets.append(dataset)
    else:
        dataset = ModelDataset(configs['data']['dataset_paths'][0], steps=configs['data']['steps'])
        info_dict = get_info_dict(configs, dataset.sampling_configs)
        amplification = TorchUtils.format(info_dict["sampling_configs"]["driver"]["amplification"])
        datasets = split_dataset_seq(dataset, configs['data']['split_percentages'])

    if info_dict['model_structure']['type'] == 'RNN':
        sequence_length = info_dict['model_structure']['sequence_length']
        chunk_size = max(configs['data']['batch_size'], sequence_length)

        start_time = time.time()
        start_memory = memory_usage()

        all_targets = []  # Collect all target values for plotting

        for i, dataset in enumerate(datasets):
            if dataset is not None:
                total_samples = len(dataset)

                input_sequences = []
                target_values = []

                buffer_data = []

                for start_idx in range(0, total_samples, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_samples)
                    chunk_data = np.array(buffer_data) if buffer_data else np.empty((0, dataset[0][0].shape[0] + 1))

                    for j in range(start_idx, end_idx):
                        sample, target = dataset[j]
                        sample = sample.cpu().numpy() if sample.is_cuda else sample.numpy()
                        target = target.cpu().numpy() if target.is_cuda else target.numpy()
                        chunk_data = np.vstack((chunk_data, np.hstack((sample, target))))

                    print(f"Number of zeros in chunk_data before sequence preparation: {np.sum(chunk_data == 0)}")

                    X, y = prepare_rnn_sequences(chunk_data, sequence_length)

                    print(f"Number of zeros in prepared input sequences X: {np.sum(X == 0)}")
                    print(f"Number of zeros in prepared target values y: {np.sum(y == 0)}")

                    input_sequences.extend(X)
                    target_values.extend(y)

                    buffer_data = chunk_data[-sequence_length + 1:].tolist() if len(chunk_data) >= sequence_length else chunk_data.tolist()

                if len(buffer_data) >= sequence_length:
                    buffer_data = np.array(buffer_data)
                    X, y = prepare_rnn_sequences(buffer_data, sequence_length)
                    input_sequences.extend(X)
                    target_values.extend(y)

                datasets[i] = [(torch.tensor(x).float(), torch.tensor(y_).float()) for x, y_ in zip(input_sequences, target_values)]

                all_targets.extend(target_values)  # Collect target values

        plot_targets(np.array(all_targets), filename="rnn_prepared_targets.png", title="RNN Prepared Target Values", xlabel="Sequence Index", ylabel="Target Value")

        for i, dataset in enumerate(datasets):
            if dataset is not None:
                print(f"Number of zeros in dataset[{i}] inputs: {sum([torch.sum(x == 0).item() for x, _ in dataset])}")
                print(f"Number of zeros in dataset[{i}] targets: {sum([torch.sum(y == 0).item() for _, y in dataset])}")

        end_time = time.time()
        end_memory = memory_usage()

        print(f"Processing Time: {end_time - start_time} seconds")
        print(f"Memory usage: {end_memory - start_memory} MB")

    dataloaders = []
    shuffle = [False, False, False]
    for i in range(len(datasets)):
        if datasets[i] is not None and len(datasets[i]) != 0:
            dl = DataLoader(
                dataset=datasets[i],
                batch_size=configs["data"]["batch_size"],
                num_workers=configs["data"]["worker_no"],
                pin_memory=configs["data"]["pin_memory"],
                shuffle=shuffle[i],
            )
            dl.tag = dataset_names[i]
            dataloaders.append(dl)
        else:
            dataloaders.append(None)
    return dataloaders, amplification, info_dict

def split_dataset(dataset, split_percentages):
    assert sum(split_percentages) == 1, "Split percentages should add to one"
    assert 1 <= len(split_percentages) <= 3, "Split percentage list should only allow from one to three values. For training, validation, and test datasets."
    train_set_size = math.ceil(split_percentages[0] * len(dataset))
    valid_set_size = math.floor(split_percentages[1] * len(dataset)) if len(split_percentages) > 1 else 0
    test_set_size = len(dataset) - train_set_size - valid_set_size

    if len(split_percentages) == 1:
        return [dataset, None, None]
    elif len(split_percentages) == 2:
        return list(random_split(dataset, [train_set_size, valid_set_size]))
    else:
        return list(random_split(dataset, [train_set_size, valid_set_size, test_set_size]))

def split_dataset_seq(dataset, split_percentages):
    assert sum(split_percentages) == 1, "Split percentages should add to one"
    assert 1 <= len(split_percentages) <= 3, "Split percentage list should only allow from one to three values. For training, validation, and test datasets."
    
    total_len = len(dataset)
    train_set_size = math.ceil(split_percentages[0] * total_len)
    valid_set_size = math.floor(split_percentages[1] * total_len) if len(split_percentages) > 1 else 0
    test_set_size = total_len - train_set_size - valid_set_size
    
    indices = list(range(total_len))
    
    train_indices = indices[:train_set_size]
    valid_indices = indices[train_set_size:train_set_size + valid_set_size]
    test_indices = indices[train_set_size + valid_set_size:]
    
    train_subset = Subset(dataset, train_indices)
    valid_subset = Subset(dataset, valid_indices) if valid_set_size > 0 else None
    test_subset = Subset(dataset, test_indices) if test_set_size > 0 else None
    
    return [train_subset, valid_subset, test_subset]

def prepare_rnn_sequences(data, sequence_length):
    input_sequences = []
    target_values = []

    for start_idx in range(len(data) - sequence_length):
        input_seq = data[start_idx:start_idx + sequence_length, :-1]
        target_value = data[start_idx + sequence_length - 1, -1]
        input_sequences.append(input_seq)
        target_values.append(target_value)

    print(f"Number of zeros in input sequences: {np.sum(input_sequences == 0)}")
    print(f"Number of zeros in target values: {np.sum(target_values == 0)}")
    return input_sequences, target_values

def memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 ** 2 

def plot_targets(targets: torch.Tensor, filename: str = "plot_ttargets.png", title: str = "Targets Plot", xlabel: str = "Sample Index", ylabel: str = "Target Value") -> None:
    if isinstance(targets, torch.Tensor):
        print(f"Targets is a tensor. Device: {targets.device}")
    
        if targets.is_cuda:
            targets = targets.cpu()
            print("Moved targets to CPU.")
        
        targets = targets.numpy()
        print("Converted targets to numpy array.")

    print(f"Targets type: {type(targets)}, shape: {targets.shape}")

    plt.figure(figsize=(10, 6))
    plt.plot(targets, label="Target Value")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
