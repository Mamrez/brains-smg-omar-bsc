"""
File containing a class for loading sampling data as a dataset, as well as a function for loading the dataset into a PyTorch dataloader.
"""
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

os.environ["CUDA_VISIBLE_DEVICES"]="2"
class ModelDataset(Dataset):
    def __init__(self, filename: str, steps: int = 1) -> None:
        """
        Initialisation of the dataset. It loads a posprocessed_data.npz file into memory.
        The targets of this file are divided by the amplification correction factor, so that
        data is made setup independent.

        Parameters
        ----------
        filename : str
            Folder and filename where the posprocessed_data.npz is.
        steps : int
            It allows to skip parts of the data when loading it into memory. The number indicates
            how many items will be skipped in between. By default, step number is one (no values
            are skipped). E.g., if steps = 2, and the inputs are [0, 1, 2, 3, 4, 5, 6]. The only
            inputs taken into account would be: [0, 2, 4, 6].

        Notes
        -----
        The postprocessed data is a .npz file called postprocessed_data.npz
        with keys: inputs, outputs and info (dict)

        1. inputs: np.array
        The input(s) is(are) gathered for all activation electrodes. The units is in Volts.

        2. outputs: The output(s) is(are) gathered from all the readout electrodes. The units are in nA.
        The output data is raw. Additional amplification correction might be needed, this is
        left for the user to decide.

        3. info: dict
        Data structure of output and input are arrays of NxD, where N is the number of samples
        and D is the dimension.

        The configs dictionary contains a copy of the configurations used for sampling the data.
        In addition, the configs dictionary has a key named electrode_info, which is created
        during the postprocessing step. The electrode_info key contains the following keys:
        3.1 electrode_no: int
        Total number of electrodes in the device

        3.2 activation_electrodes: dict

        3.2.1 electrode_no: int
        Number of activation electrodes used for gathering the data

        3.2.2 voltage_ranges: list
        Voltage ranges used for gathering the data. It contains the ranges per
        electrode, where the shape is (electrode_no,2). Being 2 the minimum and
        maximum of the ranges, respectively.

        3.3 output_electrodes: dict
        
        3.3.1 electrode_no : int
        Number of output electrodes used for gathering the data

        3.3.2 clipping_value: list[float,float]
        Value used to apply a clipping to the sampling data within the specified
        values.

        3.3.3 amplification: float
        Amplification correction factor used in the device to correct the
        amplification applied to the output current in order to convert it into
        voltage before its readout.
        """
        self.inputs, targets, self.sampling_configs = self.load_data_from_npz(
            filename, steps)
        self.targets = (targets /
                        self.sampling_configs["driver"]["amplification"])
        self.inputs = TorchUtils.format(self.inputs)
        self.targets = TorchUtils.format(self.targets)

        assert len(self.inputs) == len(
            self.targets), "Inputs and Outpus have NOT the same length"

    def __len__(self) -> int:
        """
        Overwrittes the __len__ method from the super class torch.utils.data.

        Returns
        -------
        int
            Size of the whole dataset.
        """
        return len(self.inputs)

    def __getitem__(self, index: int) -> Tuple[np.array]:
        """
        Overwrittes the __getitem__ method from the super class torch.utils.data.
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
        return (self.inputs[index, :], self.targets[index, :])

    def load_data_from_npz(self, filename: str,
                           steps: int) -> Tuple[np.array, np.array, dict]:
        """
        Loads the inputs, targets and sampling configurations from a given postprocessed_data.npz
        file.

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
            Dictionary containing the sampling configurations with which the data was
            acquired.

        Notes
        -----
        The postprocessed data is a .npz file called postprocessed_data.npz
        with keys: inputs, outputs and info (dict)

        1. inputs: np.array
        The input(s) is(are) gathered for all activation electrodes. The units is in Volts.

        2. outputs: The output(s) is(are) gathered from all the readout electrodes. The units are in nA.
        The output data is raw. Additional amplification correction might be needed, this is
        left for the user to decide.

        3. info: dict
        Data structure of output and input are arrays of NxD, where N is the number of samples
        and D is the dimension.

        The configs dictionary contains a copy of the configurations used for sampling the data.
        In addition, the configs dictionary has a key named electrode_info, which is created
        during the postprocessing step. The electrode_info key contains the following keys:
        3.1 electrode_no: int
        Total number of electrodes in the device

        3.2 activation_electrodes: dict

        3.2.1 electrode_no: int
        Number of activation electrodes used for gathering the data

        3.2.2 voltage_ranges: list
        Voltage ranges used for gathering the data. It contains the ranges per
        electrode, where the shape is (electrode_no,2). Being 2 the minimum and
        maximum of the ranges, respectively.

        3.3 output_electrodes: dict
        
        3.3.1 electrode_no : int
        Number of output electrodes used for gathering the data

        3.3.2 clipping_value: list[float,float]
        Value used to apply a clipping to the sampling data within the specified
        values.

        3.3.3 amplification: float
        Amplification correction factor used in the device to correct the
        amplification applied to the output current in order to convert it into
        voltage before its readout.
        """
        print("\n* Loading data from file:\n" + filename)
        # Pickle = True, since it also contains a dictionary.
        with np.load(filename, allow_pickle=True) as data:
            sampling_configs = dict(data["sampling_configs"].tolist())
            inputs = data["inputs"][::steps]
            outputs = data["outputs"][::steps]
            print(
                f"\t- Shape of inputs:  {inputs.shape}\n\t- Shape of outputs: {outputs.shape}\n"
            )
            print(
                f"* Sampling configs has the following keys:\n\t{sampling_configs.keys()}\n"
            )
        return inputs, outputs, sampling_configs


def get_info_dict(training_configs: dict, sampling_configs: dict) -> dict:
    """
    Retrieve the info dictionary given the training configs and the sampling configs.
    Note that the electrode_info key should be present in the sampling configs. This
    key is automatically generated when postprocessing the data.

    Parameters
    ----------
    training_configs : dict
        A copy of the configurations used for training the surrogate model.
    sampling_configs : dict
        A copy of the configurations used for sampling the training data.

    Returns
    -------
    info_dict
        This dictionary is required in order to initialise a surrogate
        model. It contains the following keys:
        1. model_structure: dict
        The definition of the internal structure of the surrogate model, which is typically five
        fully-connected layers of 90 nodes each.

        1.1 hidden_sizes : list
        A list containing the number of nodes of each layer of the surrogate model.
        E.g., [90,90,90,90,90]
        
        1.2 D_in: int
        Number of input features of the surrogate model structure. It should correspond to
        the activation electrode number.

        1.3 D_out: int
        Number of output features of the surrogate model structure. It should correspond to
        the readout electrode number.

        2. electrode_info: dict
        It contains all the information required for the surrogate model about the electrodes.

        2.1 electrode_no: int
        Total number of electrodes in the device

        2.2 activation_electrodes: dict

        2.2.1 electrode_no: int
        Number of activation electrodes used for gathering the data

        2.2.2 voltage_ranges: list
        Voltage ranges used for gathering the data. It contains the ranges per
        electrode, where the shape is (electrode_no,2). Being 2 the minimum and
        maximum of the ranges, respectively.

        2.3 output_electrodes: dict

        2.3.1 electrode_no : int
        Number of output electrodes used for gathering the data

        2.3.2 clipping_value: list[float,float]
        Value used to apply a clipping to the sampling data within the specified
        values.

        2.3.3 amplification: float
        Amplification correction factor used in the device to correct the
        amplification applied to the output current in order to convert it into
        voltage before its readout.

        3. training_configs: dict
        A copy of the configurations used for training the surrogate model.

        4. sampling_configs : dict
        A copy of the configurations used for gathering the training data.
    """
    info_dict = {}
    info_dict["model_structure"] = training_configs["model_structure"].copy()
    info_dict["electrode_info"] = sampling_configs["electrode_info"].copy()
    del training_configs["model_structure"]
    info_dict["training_configs"] = training_configs.copy()
    del sampling_configs["electrode_info"]
    info_dict["sampling_configs"] = sampling_configs.copy()
    return info_dict


def get_dataloaders(
        configs: dict
) -> Tuple[List[torch.utils.data.DataLoader], float, dict]:
    """
    Loads all the datasets specified in the dataset_paths list key of the configurations dictionary
    and creates a dataloader.

    Parameters
    ----------
    configs : dict
        Surrogate model generation configurations.

        1. results_base_dir: str
        Directory where the trained model and corresponding performance plots will be stored.

        2. seed: int
        Sets the seed for generating random numbers to a non-deterministic random number.

        3. hyperparameters:
        epochs: int
        learning_rate: float

        4. model_structure: dict
        The definition of the internal structure of the surrogate model, which is typically five
        fully-connected layers of 90 nodes each.

        4.1 hidden_sizes : list
        A list containing the number of nodes of each layer of the surrogate model.
        E.g., [90,90,90,90,90]

        4.2 D_in: int
        Number of input features of the surrogate model structure. It should correspond to
        the activation electrode number.

        4.3 D_out: int
        Number of output features of the surrogate model structure. It should correspond to
        the readout electrode number.

        5. data:
        5.1 dataset_paths: list[str]
        A list of paths to the Training, Validation and Test datasets, stored as
        postprocessed_data.npz. It also supports adding a single training dataset, and splitting
        it using the configuration split_percentages.

        5.2 split_percentages: list[float] (Optional)
        When provided together a single dataset path, in the dataset_paths list, this variable
        allows to split it into training, validation and test datasets by providing the split
        percentage values. E.g. [0.8, 0.2] will split the training dataset into 80% of the data
        for training and 20% of the data for validation. Similarly, [0.8, 0.1, 0.1] will split
        the training dataset into 80%, 10% for validation dataset and 10% for test dataset. Note
        that all split values in the list should add to 1.

        5.3 steps : int
        It allows to skip parts of the data when loading it into memory. The number indicates
        how many items will be skipped in between. By default, step number is one (no values
        are skipped). E.g., if steps = 2, and the inputs are [0, 1, 2, 3, 4, 5, 6]. The only
        inputs taken into account would be: [0, 2, 4, 6].

        5.4 batch_size: int
        How many samples will contain each forward pass.

        5.5 worker_no: int
        How many subprocesses to use for data loading. 0 means that the data will be loaded in
        the main process. (default: 0)

        5.6 pin_memory: boolean
        If True, the data loader will copy Tensors into CUDA pinned memory before returning
        them. If your data elements are a custom type, or your collate_fn returns a batch that
        is a custom type.

    Returns
    -------
    dataloaders : list[torch.utils.dataDataLoader]
        A list containing the corresponding training, validation and test datasets.

    """
    # Load dataset
    # Only training configs will be taken into account for info dict
    # For ranges and etc.
    assert 'data' in configs and 'dataset_paths' in configs['data']
    assert isinstance(configs['data']['dataset_paths'],
                      list), "Paths for datasets should be passed as a list"
    assert configs['data']['dataset_paths'] != [], "Empty paths for datasets"
    datasets = []
    info_dict = None
    amplification = None
    dataset_names = ['train', 'validation', 'test']


    if len(configs['data']['dataset_paths']) > 1:
        for i in range(len(configs['data']['dataset_paths'])):
            if configs['data']['dataset_paths'][i] is not None:
                dataset = ModelDataset(configs['data']['dataset_paths'][i],
                                       steps=configs['data']['steps'])

                if i > 0:
                    amplification_aux = TorchUtils.format(
                        info_dict["sampling_configs"]["driver"]
                        ["amplification"])
                    assert torch.eq(amplification_aux, amplification).all(), (
                        "Amplification correction factor should be the same for all datasets."
                        + "Check if all datasets come from the same setup.")
                    info_dict[dataset_names[i] +
                              '_sampling_configs'] = dataset.sampling_configs
                else:
                    info_dict = get_info_dict(configs,
                                              dataset.sampling_configs)
                amplification = TorchUtils.format(
                    info_dict["sampling_configs"]["driver"]["amplification"])
                datasets.append(dataset)

    else:
        dataset = ModelDataset(configs['data']['dataset_paths'][0],
                               steps=configs['data']['steps'])
        info_dict = get_info_dict(configs, dataset.sampling_configs)
        amplification = TorchUtils.format(
            info_dict["sampling_configs"]["driver"]["amplification"])
        datasets = split_dataset_seq(dataset, configs['data']['split_percentages'])

    
    if info_dict['model_structure']['type'] == 'RNN':
        sequence_length = info_dict['model_structure']['sequence_length']
        chunk_size = max(configs['data']['batch_size'], sequence_length)  # Ensure chunk size is at least sequence_length

        start_time = time.time()
        start_memory = memory_usage()

        for i, dataset in enumerate(datasets):
            if dataset is not None:
                total_samples = len(dataset)
                num_sequences = (total_samples // sequence_length) * sequence_length

                input_sequences = np.zeros((num_sequences, sequence_length, dataset[0][0].shape[0]))
                target_values = np.zeros(num_sequences)
                current_sequence_index = 0

                # Initialize buffer to handle sequence spanning across chunks
                buffer_data = []

                for start_idx in range(0, total_samples, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_samples)
                    chunk_data = np.array(buffer_data) if buffer_data else np.empty((0, dataset[0][0].shape[0] + 1))  # Start with any leftover data from previous chunk

                    # Collect data for the current chunk
                    for j in range(start_idx, end_idx):
                        sample, target = dataset[j]
                        sample = sample.cpu().numpy() if sample.is_cuda else sample.numpy()
                        target = target.cpu().numpy() if target.is_cuda else target.numpy()
                        chunk_data = np.vstack((chunk_data, np.hstack((sample, target))))

                    # Prepare sequences within the current chunk
                    X, y = prepare_rnn_sequences(chunk_data, sequence_length)

                    if (len(X) < len(y)):
                        continue

                    input_sequences[current_sequence_index:current_sequence_index + len(X)] = X
                    target_values[current_sequence_index:current_sequence_index + len(X)] = y
                    current_sequence_index += len(X)

                    # Save the last 'sequence_length' data points to the buffer for the next chunk
                    buffer_data = chunk_data[-sequence_length + 1:].tolist() if len(chunk_data) >= sequence_length else chunk_data.tolist()

                # Ensure any remaining data in the buffer is processed
                if len(buffer_data) >= sequence_length:
                    buffer_data = np.array(buffer_data)
                    X, y = prepare_rnn_sequences(chunk_data, sequence_length)
                    input_sequences[current_sequence_index:current_sequence_index + len(X)] = X
                    target_values[current_sequence_index:current_sequence_index + len(X)] = y

                # datasets[i] = [(x, y_) for x, y_ in zip(input_sequences, target_values)]
                datasets[i] = [(torch.tensor(x).float(), torch.tensor(y_).float()) for x, y_ in zip(input_sequences, target_values)]

        end_time = time.time()
        end_memory = memory_usage()

        print(f"Processing Time: {end_time - start_time} seconds")
        print(f"Memory usage: {end_memory - start_memory} MB")
        #         if info_dict['model_structure']['type'] == 'LSTM':
        # sequence_length = info_dict['model_structure']['sequence_length']
        # chunk_size = max(configs['data']['batch_size'], sequence_length)  # Ensure chunk size is at least sequence_length
        # for i, dataset in enumerate(datasets):
        #     if dataset is not None:
        #         total_samples = len(dataset)
        #         input_sequences, target_values = [], []

        #         # Initialize buffer to handle sequence spanning across chunks
        #         buffer_data = []

        #         for start_idx in range(0, total_samples, chunk_size):
        #             end_idx = min(start_idx + chunk_size, total_samples)
        #             chunk_data = buffer_data  # Start with any leftover data from previous chunk

        #             # Collect data for the current chunk
        #             for j in range(start_idx, end_idx):
        #                 sample, target = dataset[j]
        #                 sample = sample.cpu().numpy() if sample.is_cuda else sample.numpy()
        #                 target = target.cpu().numpy() if target.is_cuda else target.numpy()
        #                 chunk_data.append(np.hstack((sample, target)))

        #             chunk_data = np.array(chunk_data)

        #             # Prepare sequences within the current chunk
        #             X, y = prepare_rnn_sequences(chunk_data, sequence_length)

        #             input_sequences.extend(X)
        #             target_values.extend(y)

        #             # Save the last 'sequence_length' data points to the buffer for the next chunk
        #             buffer_data = chunk_data[-sequence_length + 1:].tolist()

        #         # Ensure any remaining data in the buffer is processed
        #         if len(buffer_data) >= sequence_length:
        #             buffer_data = np.array(buffer_data)
        #             X, y = prepare_rnn_sequences(buffer_data, sequence_length)
        #             input_sequences.extend(X)
        #             target_values.extend(y)

        #         datasets[i] = [(x, y_) for x, y_ in zip(input_sequences, target_values)]

    # Create dataloaders
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
    assert len(split_percentages) <= 3 and len(split_percentages) >= 1, "Split percentage list should only allow from one to three values. For training, validation datasets."
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
    input_sequences = np.zeros((len(data) - sequence_length, sequence_length, data.shape[1] - 1))
    target_values = np.zeros(len(data) - sequence_length)

    for start_idx in range(len(data) - sequence_length):
        input_seq = data[start_idx:start_idx + sequence_length, :-1]
        target_value = data[start_idx + sequence_length - 1, -1]
        input_sequences[start_idx] = input_seq
        target_values[start_idx] = target_value
    return input_sequences, target_values

def memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss /1024 **2 

# def prepare_rnn_sequences(data, sequence_length):
#     input_sequences = np.zeros((len(data) - sequence_length, sequence_length, data.shape[]))
#     input_sequences, target_values = [], []
#     for start_idx in range(len(data) - sequence_length):
#         input_seq = data[start_idx:start_idx + sequence_length, :-1]
#         target_value = data[start_idx + sequence_length - 1, -1]
#         input_sequences.append(input_seq)
#         target_values.append(target_value)
#     return np.array(input_sequences), np.array(target_values)