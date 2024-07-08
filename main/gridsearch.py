import sys
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import itertools
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import glob
from bspysmg.model.training import generate_surrogate_model
from bspysmg.data.dataset import get_dataloaders
from bspysmg.utils.plots import plot_wave_prediction, plot_error_hist, plot_error_vs_output
from bspysmg.model.lstm import LSTMModel
from bspysmg.model.gru import GRUModel


def read_yaml(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict


def save_yaml(config, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(config, file)


def train_model(config_path,custom_model):
    config = read_yaml(config_path)
    try:
       
        generate_surrogate_model(config, custom_model=custom_model, main_folder=os.path.splitext(os.path.basename(config_path))[0])
        return True, config_path
    except Exception as e:
        print(f"Error training model with config {config_path}: {e}")
        return False, config_path


def main(gridsearch_path, model_name,custom_model):
  
    print(f"Loading configuration from {gridsearch_path}")
    config = read_yaml(gridsearch_path)
    
    # Extract the base configuration and hyperparameter ranges
    base_config = config.copy()
    param_ranges = config.pop('hyperparameter_ranges')
    results_base_dir = base_config['results_base_dir']
    config_base_dir = base_config['config_base_dir']
    os.makedirs(results_base_dir, exist_ok=True)
    os.makedirs(config_base_dir, exist_ok=True)
    
    # Prepare to log results
    results = []
    param_names = list(param_ranges.keys())
    
    # Convert the range dictionary values into lists
    param_ranges_converted = {}
    for param, range_spec in param_ranges.items():
        if isinstance(range_spec, dict) and 'start' in range_spec and 'stop' in range_spec and 'step' in range_spec:
            param_ranges_converted[param] = list(range(range_spec['start'], range_spec['stop'], range_spec['step']))
        elif isinstance(range_spec, dict) and 'values' in range_spec:
            param_ranges_converted[param] = range_spec['values']
        else:
            raise ValueError(f"Invalid range specification for parameter: {param}")

    param_combinations = itertools.product(*param_ranges_converted.values())

    try:
        # Iterate over all combinations of hyperparameters
        for idx, combination in enumerate(param_combinations):
            # Create a clean copy of the base configuration
            config_to_save = {
                'results_base_dir': base_config['results_base_dir'],
                'model_structure': base_config['model_structure'].copy(),
                'hyperparameters': base_config['hyperparameters'].copy(),
                'data': base_config['data'].copy()
            }

            # Update the configuration with the current combination of hyperparameters
            num_layers = None
            for param, value in zip(param_names, combination):
                keys = param.split('.')
                sub_config = config_to_save
                for key in keys[:-1]:
                    sub_config = sub_config[key]
                sub_config[keys[-1]] = value
                if param == 'model_structure.num_layers':
                    num_layers = value
            
            # Define the filename for the current configuration
            config_filename = f"config_{model_name}_L_{num_layers}_ID_{idx}.yaml"
            config_path = os.path.join(config_base_dir, config_filename)
            
            # Save the current configuration to a YAML file
            save_yaml(config_to_save, config_path)
            
            # Train the model with the current configuration
            success, config_path = train_model(config_path,custom_model=custom_model)
            if success:
                # If training is successful, retrieve losses from saved model
                result = {'config': config_filename}
                result.update({param: value for param, value in zip(param_names, combination)})

                # Find the directory containing the training data
                training_dirs = glob.glob(os.path.join(results_base_dir, f"*L_{num_layers}_ID_{idx}*"))
                if training_dirs:
                    latest_training_dir = max(training_dirs, key=os.path.getmtime)
                    training_data_path = os.path.join(latest_training_dir, "training_data.pt")
                    training_data = torch.load(training_data_path)
                    result['train_loss'] = training_data['train_losses'][-1].item()
                    result['val_loss'] = training_data['val_losses'][-1].item()
                    results.append(result)
    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:

        results_df = pd.DataFrame(results)
        
        # Save results to a CSV file
        results_df.to_csv(os.path.join(results_base_dir, 'results.csv'), index=False)
        
        # Print out the ID of the lowest validation loss model
        if not results_df.empty:
            best_model_id = results_df.loc[results_df['val_loss'].idxmin()]['config']
            print(f"The model with the lowest validation loss is: {best_model_id}")


if __name__ == "__main__":
    main("configs\gridsearch\gridsearch.yaml", "GRU_RestTime_layer2",custom_model=GRUModel)
    