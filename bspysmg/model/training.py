"""
File containing functions for training a surrogate model in pytorch taking into account the error in nano amperes.
"""
import os
import torch
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import gc
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import MSELoss
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.io import create_directory_timestamp
from brainspy.processors.simulation.model import NeuralNetworkModel
from bspysmg.data.dataset import get_dataloaders
from bspysmg.utils.early_stopping import EarlyStopping
from bspysmg.utils.plots import plot_error_vs_output, plot_error_hist, plot_wave_prediction
from bspysmg.model.lstm import LSTMModel
from bspysmg.model.gru import GRUModel
from typing import Tuple, List
from bspysmg.model.xgboost import XGBoostModel
import xgboost as xgb
from bspysmg.model.esn import ESNModel

def init_seed(configs: dict) -> None:
    """
    Initializes a random seed for training. A random seed is a starting point for pseudorandom
    number generator algorithms which is used for reproducibility.
    Also see - https://pytorch.org/docs/stable/notes/randomness.html  

    Parameters
    ----------
    configs : dict
        Training configurations with the following keys:

        1. seed:  int [Optional]
        The desired seed for the random number generator. If the dictionary does not contain 
        this key, a deterministic random seed will be applied, and added to the key 'seed' in 
        the dictionary.
    """
    if "seed" in configs:
        seed = configs["seed"]
    else:
        seed = None

    seed = TorchUtils.init_seed(seed, deterministic=True)
    configs["seed"] = seed


def generate_surrogate_model(
        configs: dict,
        custom_model=None,
        criterion: torch.nn.modules.loss._Loss = MSELoss(),
        custom_optimizer: torch.optim.Optimizer = Adam,
        main_folder: str = "training_data") -> None:

    init_seed(configs)
    results_dir = create_directory_timestamp(configs["results_base_dir"], main_folder)

    dataloaders, amplification, info_dict = get_dataloaders(configs)

    if custom_model is None:
        raise ValueError("custom_model must be provided and cannot be None")


    if isinstance(custom_model, type) and issubclass(custom_model, XGBoostModel):
        model = custom_model(info_dict["model_structure"], dataloaders[0], dataloaders[1])
        saved_dir = results_dir
    elif isinstance(custom_model, type) and issubclass(custom_model, ESNModel):
        model = custom_model(info_dict["model_structure"],dataloaders)
        saved_dir = results_dir

    else:
        model = custom_model(info_dict["model_structure"])
        model = TorchUtils.format(model)

        optimizer = custom_optimizer(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=configs["hyperparameters"]["learning_rate"],
            betas=(0.9, 0.75),
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {total_params}")
        
        scaler = GradScaler()

        model, performances, saved_dir = train_loop(
            model,
            info_dict,
            (dataloaders[0], dataloaders[1]),
            criterion,
            optimizer,
            configs["hyperparameters"]["epochs"],
            amplification,
            save_dir=results_dir,
            scaler=scaler
        )

    # Plot results
    labels = ["TRAINING", "VALIDATION", "TEST"]
    for i in range(len(dataloaders)):
        if dataloaders[i] is not None:
            io_file_path = 'main/mainSamplingData/IO.dat'
            loss = postprocess(
                dataloaders[i],
                model,
                criterion,
                amplification,
                results_dir,
                label=labels[i],
                io_file_path=io_file_path,
                start_index=start_index
            )

    if not issubclass(custom_model, XGBoostModel) and not issubclass(custom_model, ESNModel):
        plt.figure()
        plt.plot(TorchUtils.to_numpy(performances[0]))
        if len(performances) > 1 and not len(performances[1]) == 0:
            plt.plot(TorchUtils.to_numpy(performances[1]))
        if dataloaders[-1].tag == 'test':
            plt.plot(np.ones(len(performances[-1])) * TorchUtils.to_numpy(loss))
            plt.title("Training profile /n Test loss : %.6f (nA)" % loss)
        else:
            plt.title("Training profile")
        if not len(performances[1]) == 0:
            plt.legend(["training", "validation"])
        plt.xlabel("Epoch no.")
        plt.ylabel("RMSE (nA)")
        plt.savefig(os.path.join(results_dir, "training_profile"))
        if not dataloaders[-1].tag == 'train':
            training_data = torch.load(
                os.path.join(results_dir, "training_data.pt"))
            training_data['test_loss'] = loss
            torch.save(training_data, os.path.join(results_dir,
                                                "training_data.pt"))
    return saved_dir



def train_loop(
    model: torch.nn.Module,
    info_dict: dict,
    dataloaders: List[torch.utils.data.DataLoader],
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    amplification: float,
    scaler: GradScaler, 
    start_epoch: int = 0,
    save_dir: str = None,
    early_stopping: bool = True,
    patience: int = 5,
    accumulation_steps: int = 1

) -> Tuple[torch.nn.Module, List[float]]:
    """
    Performs the training of a model and returns the trained model, training loss
    validation loss. It also saves the model in each epoch if current validation
    loss is less than the previous validation loss.

    Parameters
    ----------
    model : custom model of type torch.nn.Module
        Model to be trained.
    info_dict : dict
        The dictionary used for initialising the surrogate model. It has the following keys:
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
    dataloaders :  list
        A list containing a single PyTorch Dataloader containing the training dataset.
    criterion : <method>
        Loss function that will be used to train the model.
    optimizer : torch.optim.Optimizer
        Optimization method used to train the model which decreases model's loss.
    epochs : int
        The number of iterations for which the model is to be trained.
    amplification: float
        Amplification correction factor used in the device to correct the amplification
        applied to the output current in order to convert it into voltage before its
        readout.
    start_epoch : int [Optional]
        The starting value of the epochs.
    save_dir : string [Optional]
        Name of the path and file where the trained model is to be saved.
    early_stopping : bool [Optional]
        If this is set to true, early stopping algorithm is used during the training
        of the model.
        Also see - https://medium.com/analytics-vidhya/early-stopping-with-pytorch-to-
        restrain-your-model-from-overfitting-dce6de4081c5

    Returns
    -------
    model
        Trained model
    losses
        list of training loss and validation loss.
    saved_dir
        directory where the model was saved
    """
    if start_epoch > 0:
        start_epoch += 1

    train_losses, val_losses = TorchUtils.format([]), TorchUtils.format([])
    min_val_loss = np.inf
    early_stopper = EarlyStopping(patience=patience, verbose=True, path=os.path.join(save_dir, 'checkpoint.pt'))



    for epoch in range(epochs):
        print("\nEpoch: " + str(epoch))
        model, running_loss = default_train_step(model, dataloaders[0],
                                                 criterion, optimizer, scaler=scaler)
        running_loss = running_loss**(1 / 2)
        running_loss *= amplification
        train_losses = torch.cat((train_losses, running_loss.unsqueeze(dim=0)),
                                 dim=0)
        description = "Training loss (RMSE): {:.6f} (nA)\n".format(
            train_losses[-1].item())

        if dataloaders[1] is not None and len(dataloaders[1]) > 0:
            val_loss = default_val_step(model, dataloaders[1], criterion,scaler=scaler)
            val_loss = val_loss**(1 / 2)
            val_loss *= amplification
            val_losses = torch.cat((val_losses, val_loss.unsqueeze(dim=0)),
                                   dim=0)
            description += "Validation loss (RMSE): {:.6f} (nA)\n".format(
                val_losses[-1].item())
            
            
            # Save only when peak val performance is reached
            if (save_dir is not None and early_stopping
                    and val_losses[-1] < min_val_loss):
                min_val_loss = val_losses[-1]
                description += "Model saved in: " + save_dir
                # torch.save(model, os.path.join(save_dir, "model.pt"))
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "info": info_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "min_val_loss": min_val_loss,
                    },
                    os.path.join(save_dir, "training_data.pt"),
                )

        print(description)
        # looper.set_description(description)

        # TODO: Add a save instruction and a stopping criteria
        # if stopping_criteria(train_losses, val_losses):
        #     break
        early_stopper(val_loss.item(), model)

        if early_stopper.early_stop:
            print('Early stopping')
            break
    
    print("\nFinished training model. ")
    print("Model saved in: " + save_dir)
    if (save_dir is not None and early_stopping and dataloaders[1] is not None
            and len(dataloaders[1]) > 0):
        training_data = torch.load(os.path.join(save_dir, "training_data.pt"))
        model.load_state_dict(training_data["model_state_dict"])
        print("Min validation loss (RMSE): {:.6f} (nA)\n".format(
            min_val_loss.item()))
    else:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "info": info_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "min_val_loss": min_val_loss,
            },
            os.path.join(save_dir, "training_data.pt"),
        )

    return model, [train_losses, val_losses], save_dir


def train_loop_xgboost(
    model,
    info_dict: dict,
    dataloaders: List[torch.utils.data.DataLoader],
    amplification: float,
    save_dir: str = None
) -> Tuple[xgb.XGBRegressor, List[float]]:
    train_losses, val_losses = [], []
    
    # Prepare the training data
    X_train, y_train = [], []
    for inputs, targets in dataloaders[0]:
        inputs = inputs.view(inputs.size(0), -1).numpy()  # Flattening the sequences into 2D
        targets = targets.numpy()
        X_train.append(inputs)
        y_train.append(targets)

    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train).reshape(-1)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Calculate the training loss
    train_predictions = model.predict(X_train)
    train_loss = np.sqrt(np.mean((train_predictions - y_train) ** 2)) * amplification
    train_losses.append(train_loss)
    print("Training loss (RMSE): {:.6f} (nA)".format(train_losses[-1].item()))

    # Prepare the validation data
    if dataloaders[1] is not None and len(dataloaders[1]) > 0:
        X_val, y_val = [], []
        for inputs, targets in dataloaders[1]:
            inputs = inputs.view(inputs.size(0), -1).numpy()  # Flattening the sequences into 2D
            targets = targets.numpy()
            X_val.append(inputs)
            y_val.append(targets)

        X_val = np.vstack(X_val)
        y_val = np.hstack(y_val).reshape(-1)

        # Calculate the validation loss
        val_predictions = model.predict(X_val)
        val_loss = np.sqrt(np.mean((val_predictions - y_val) ** 2)) * amplification
        val_losses.append(val_loss)
        print("Validation loss (RMSE): {:.6f} (nA)".format(val_losses[-1].item()))

        # Save the model if a save directory is provided
        if save_dir is not None:
            model.model.save_model(os.path.join(save_dir, "model.xgb"))
            print("Model saved in: " + save_dir)

    return model, [train_losses, val_losses], save_dir

def default_train_step(
        model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer, scaler: GradScaler, accumulation_steps: int = 5) -> Tuple[torch.nn.Module, float]:
    """
    Performs the training step of a model within a single epoch and returns the
    current loss and current trained model.

    Parameters
    ----------
    model : custom model of type torch.nn.Module
        Model to be trained.
    dataloader :  torch.utils.data.DataLoader
        A PyTorch Dataloader containing the training dataset.
    criterion : <method>
        Loss function that will be used to train the model.
    optimizer : torch.optim.Optimizer
        Optimization method used to train the model which decreases model's loss.

    Returns
    -------
    tuple
        Trained model and training loss for the current epoch.
    """
    running_loss = 0
    model.train()
    optimizer.zero_grad()
    loop = tqdm(dataloader)
    for i, (inputs, targets) in enumerate(loop):
        inputs, targets = to_device(inputs), to_device(targets)
        optimizer.zero_grad()

        if hasattr(model, 'initialize_hidden_state'):
            model.initialize_hidden_state(inputs.size(0),dtype=inputs.dtype)

        # with autocast():
        #     predictions = model(inputs)
        #     loss = criterion(predictions, targets)

        with autocast():
            predictions = model(inputs)
            if torch.isnan(predictions).any():
                print("Predictions contain nan values!")
                # print(predictions)
            loss = criterion(predictions, targets)
            if torch.isnan(loss).any():
                print("Loss contains nan values!")
                # print(loss)

        loss = loss / accumulation_steps
        scaler.scale(loss).backward()


        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * inputs.shape[0]
        clear_memory()

    running_loss /= len(dataloader.dataset)
    return model, running_loss


def default_val_step(model: torch.nn.Module,
                     dataloader: torch.utils.data.DataLoader,
                     criterion: torch.nn.modules.loss._Loss,
                     scaler: GradScaler) -> float:
    """
    Performs the validation step of a model within a single epoch and returns
    the validation loss.

    Parameters
    ----------
    model : custom model of type torch.nn.Module
        Model to be trained.
    dataloader :  torch.utils.data.DataLoader
        A PyTorch Dataloader containing the training dataset.
    criterion : <method>
        Loss function that will be used to train the model.

    Returns
    -------
    float
        Validation loss for the current epoch.
    """
    with torch.no_grad():
        val_loss = 0
        model.eval()
        loop = tqdm(dataloader)
        for inputs, targets in loop:
            inputs, targets = to_device(inputs), to_device(targets)

            # Ensure the target tensor has the same shape as the input tensor
            targets = targets.view(-1, 1)

            if hasattr(model, 'initialize_hidden_state'):
                model.initialize_hidden_state(inputs.size(0),dtype=inputs.dtype)

            with autocast():
                predictions = model(inputs)
                loss = criterion(predictions, targets)

            val_loss += loss.item() * inputs.shape[0]
            torch.cuda.empty_cache()

            loop.set_postfix(batch_loss=loss.item())
        val_loss /= len(dataloader.dataset)
    return val_loss

import xgboost as xgb
from sklearn.metrics import mean_squared_error

def postprocess(dataloader: torch.utils.data.DataLoader, model, criterion: torch.nn.modules.loss._Loss,
                amplification: float, results_dir: str, label: str, io_file_path: str = None, start_index: int = 0) -> float:
    """
    Plots error vs output and error histogram for given dataset and saves it to
    specified directory.

    Parameters
    ----------
    dataloader :  torch.utils.data.DataLoader
        A PyTorch Dataloader containing the training dataset.
    model : custom model of type torch.nn.Module or XGBoostModel
        Model to be trained.
    criterion : <method>
        Loss function that will be used to train the model.
    amplification: float
        Amplification correction factor used in the device to correct the amplification
        applied to the output current in order to convert it into voltage before its
        readout.
    results_dir : string
        Name of the path and file where the plots are to be saved.
    label : string
        Name of the dataset. I.e., train, validation or test.

    Returns
    -------
    float
        Mean Squared error evaluated on given dataset.
    """
    print(f"Postprocessing {label} data ... ")
    running_loss = 0
    all_targets = []
    all_predictions = []

    if isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader):
                inputs, targets = to_device(inputs), to_device(targets)
                targets = targets.view(-1, 1)

                predictions = model(inputs)
                all_targets.append(amplification * targets)
                all_predictions.append(amplification * predictions)
                loss = criterion(predictions, targets)
                running_loss += loss * inputs.shape[0]  # sum up batch loss

        running_loss /= len(dataloader.dataset)
        running_loss = running_loss * (amplification ** 2)

        all_targets = TorchUtils.to_numpy(torch.cat(all_targets, dim=0))
        all_predictions = TorchUtils.to_numpy(torch.cat(all_predictions, dim=0))
    
    elif isinstance(model, XGBoostModel):
        dtest = model.dataloader_to_dmatrix(dataloader)
        predictions = model.predict(dtest)
        predictions = np.array(predictions, dtype=np.float32).reshape(-1, 1)
        amplification = amplification.cpu().numpy()
        all_predictions.extend((amplification * predictions).tolist())
        for _, targets in dataloader:
            targets = targets.cpu().numpy().astype(np.float32).reshape(-1, 1)
            all_targets.extend((amplification * targets).tolist())
        
        all_targets = np.array(all_targets, dtype=np.float32).reshape(-1, 1)
        all_predictions = np.array(all_predictions, dtype=np.float32).reshape(-1, 1)
        running_loss = mean_squared_error(all_targets, all_predictions,squared=False)

    elif isinstance(model, ESNModel):
        all_targets = []
        all_predictions = []
        running_loss = 0.0

        for inputs, targets in tqdm(dataloader):
            # Move inputs and targets to CPU and convert to numpy arrays if they are not already
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.cpu().numpy()
            if isinstance(targets, torch.Tensor):
                targets = targets.cpu().numpy()
            if isinstance(amplification, torch.Tensor):
                amplification = amplification.cpu().numpy()
            # Make predictions with the ESN model
            predictions = model.predict(inputs)
            
            # Amplify the targets and predictions
            targets = amplification * targets
            predictions = amplification * predictions
            
            all_targets.append(targets)
            all_predictions.append(predictions)
            
            # Calculate the loss for the current batch
            batch_loss = mean_squared_error(targets, predictions, squared=False)
            running_loss += batch_loss * inputs.shape[0]  # Sum up batch loss
            
        # Average running loss over the entire dataset
        running_loss /= len(dataloader.dataset)
        running_loss = running_loss * (amplification ** 2)
        
        # Concatenate all targets and predictions
        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)

        # Convert running_loss to a scalar if it is a numpy array
        if isinstance(running_loss, np.ndarray):
            running_loss = running_loss.item()

    else:
        raise ValueError("Model must be of type torch.nn.Module or XGBoostModel")

    error = all_targets - all_predictions

    print(f"{label.capitalize()} loss (MSE): {running_loss:.6f} (nA)")
    print(f"{label.capitalize()} loss (RMSE): {np.sqrt(running_loss):.6f} (nA)\n")

    plot_error_vs_output(all_targets, error, results_dir, name=label + "_error_vs_output")
    plot_error_hist(all_targets, all_predictions, error, running_loss, results_dir, name=label + "_error")

    if io_file_path:
        plot_wave_prediction(io_file_path, all_predictions, data_type=label, save_directory=results_dir, start_index=start_index, all_targets=all_targets)

    try:
        np.savez(os.path.join(results_dir, f"predictionTargetsData_{label}.npz"), all_targets, all_predictions)
    except Exception as e:
        print(f"Exception occurred while saving npz file: {e}")

    return np.sqrt(running_loss)

# def postprocess(dataloader: torch.utils.data.DataLoader,
#                 model, criterion: torch.nn.modules.loss._Loss,
#                 amplification: float, results_dir: str, label: str, io_file_path: str = None, start_index: int = 0) -> float:
#     """
#     Plots error vs output and error histogram for given dataset and saves it to
#     specified directory.

#     Parameters
#     ----------
#     dataloader :  torch.utils.data.DataLoader
#         A PyTorch Dataloader containing the training dataset.
#     model : custom model of type torch.nn.Module
#         Model to be trained.
#     criterion : <method>
#         Loss function that will be used to train the model.
#     amplification: float
#         Amplification correction factor used in the device to correct the amplification
#         applied to the output current in order to convert it into voltage before its
#         readout.
#     results_dir : string
#         Name of the path and file where the plots are to be saved.
#     label : string
#         Name of the dataset. I.e., train, validation or test.

#     Returns
#     -------
#     float
#         Mean Squared error evaluated on given dataset.
#     """
#     print(f"Postprocessing {label} data ... ")
#     # i = 0
#     running_loss = 0
#     all_targets = []
#     all_predictions = []
#     with torch.no_grad():
#         model.eval()
#         for inputs, targets in tqdm(dataloader):
#             inputs, targets = to_device(inputs), to_device(targets)

#             # Ensure the target tensor has the same shape as the input tensor
#             targets = targets.view(-1, 1)

#             if isinstance(model, LSTMModel) or isinstance(model,GRUModel):
#                 model.initialize_hidden_state(inputs.size(0),inputs.dtype)


#             predictions = model(inputs)
#             all_targets.append(amplification * targets)
#             all_predictions.append(amplification * predictions)
#             loss = criterion(predictions, targets)
#             running_loss += loss * inputs.shape[0]  # sum up batch loss

#     running_loss /= len(dataloader.dataset)
#     running_loss = running_loss * (amplification**2)

#     print(label.capitalize() +
#           " loss (MSE): {:.6f} (nA)".format(running_loss.item()))
#     print(
#         label.capitalize() +
#         " loss (RMSE): {:.6f} (nA)\n".format(torch.sqrt(running_loss).item()))

#     all_targets = TorchUtils.to_numpy(torch.cat(all_targets, dim=0))
#     all_predictions = TorchUtils.to_numpy(torch.cat(all_predictions, dim=0))
    
#     error = all_targets - all_predictions
    
#     plot_error_vs_output(
#         all_targets,
#         error,
#         results_dir,
#         name=label + "_error_vs_output",
#     )
#     plot_error_hist(
#         all_targets,
#         all_predictions,
#         error,
#         TorchUtils.to_numpy(running_loss),
#         results_dir,
#         name=label + "_error",
#     )

#      # Plot wave predictions if IO file path is provided
#     if io_file_path:
#         plot_wave_prediction(io_file_path, all_predictions, data_type=label, save_directory=results_dir,start_index=start_index, all_targets=all_targets)

#     try:
#         np.savez(results_dir + f"/predictionTargetsData_{label}.npz",all_targets,all_predictions)
#     except:
#         print("Exception occured!")

#     return torch.sqrt(running_loss)

# import numpy as np
# from tqdm import tqdm
# from sklearn.metrics import mean_squared_error
# import torch

# import numpy as np
# from tqdm import tqdm
# from sklearn.metrics import mean_squared_error
# import torch

# def postprocess(
#     dataloader: torch.utils.data.DataLoader,
#     model,
#     criterion: torch.nn.modules.loss._Loss,
#     amplification: float,
#     results_dir: str,
#     label: str,
#     io_file_path: str = None,
#     start_index: int = 0
# ) -> float:
#     """
#     Plots error vs output and error histogram for given dataset and saves it to
#     specified directory.

#     Parameters
#     ----------
#     dataloader :  torch.utils.data.DataLoader
#         A PyTorch Dataloader containing the training dataset.
#     model : XGBoostModel
#         Model to be trained.
#     criterion : <method>
#         Loss function that will be used to train the model.
#     amplification: float
#         Amplification correction factor used in the device to correct the amplification
#         applied to the output current in order to convert it into voltage before its
#         readout.
#     results_dir : string
#         Name of the path and file where the plots are to be saved.
#     label : string
#         Name of the dataset. I.e., train, validation or test.

#     Returns
#     -------
#     float
#         Mean Squared error evaluated on given dataset.
#     """
#     print(f"Postprocessing {label} data ... ")
#     running_loss = 0
#     all_targets = []
#     all_predictions = []

#     # Collect all targets and predictions
#     for inputs, targets in tqdm(dataloader):
#         inputs = inputs.view(inputs.size(0), -1).cpu().numpy()  # Flattening the sequences into 2D and move to CPU
#         if targets.is_cuda:
#             targets = targets.cpu().numpy()  # Move targets to CPU if they are on GPU
#         else:
#             targets = targets.numpy()

#         predictions = model.predict(inputs)
#         all_targets.append(amplification * targets)
#         all_predictions.append(amplification * predictions)
#         loss = mean_squared_error(targets, predictions)
#         running_loss += loss * inputs.shape[0]  # sum up batch loss

#     if not all_targets:
#         raise RuntimeError("No targets found in dataloader.")

#     if not all_predictions:
#         raise RuntimeError("No predictions made by the model.")

#     running_loss /= len(dataloader.dataset)
#     running_loss = running_loss * (amplification**2)

#     print(label.capitalize() +
#           " loss (MSE): {:.6f} (nA)".format(running_loss))
#     print(label.capitalize() +
#           " loss (RMSE): {:.6f} (nA)\n".format(np.sqrt(running_loss)))

#     all_targets = np.concatenate(all_targets, axis=0)
#     all_predictions = np.concatenate(all_predictions, axis=0)
    
#     error = all_targets - all_predictions
    
#     plot_error_vs_output(
#         all_targets,
#         error,
#         results_dir,
#         name=label + "_error_vs_output",
#     )
#     plot_error_hist(
#         all_targets,
#         all_predictions,
#         error,
#         running_loss,
#         results_dir,
#         name=label + "_error",
#     )

#     if io_file_path:
#         plot_wave_prediction(io_file_path, all_predictions, data_type=label, save_directory=results_dir, start_index=start_index, all_targets=all_targets)

#     return np.sqrt(running_loss)



def to_device(inputs: torch.Tensor) -> torch.Tensor:
    """
    Copies input tensors from CPU to GPU device for processing. GPU allows multithreading
    which makes computation faster. The rule of thumb is using 4 worker threads per GPU.
    See - https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device

    Parameters
    ----------
    inputs : torch.Tensor
        Input tensor which needs to be loaded into GPU device.

    Returns
    -------
    torch.Tensor
        Input tensor allocated to GPU device.
    """
    if inputs.device != TorchUtils.get_device():
        inputs = inputs.to(device=TorchUtils.get_device()).float()
    return inputs

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()