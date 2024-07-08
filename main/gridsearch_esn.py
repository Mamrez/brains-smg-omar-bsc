import sys
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import nrmse, rsquare
from reservoirpy.hyper import research, plot_hyperopt_report
from bspysmg.data.dataset import get_dataloaders

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict

def save_yaml(config, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(config, file)

def process_hyperopt_config(hyperopt_config):
    hp_space = hyperopt_config.get('hp_space', {})
    for key, value in hp_space.items():
        # Convert all elements except the first one to float first
        hp_space[key] = [value[0]] + [float(v) for v in value[1:]]
        # Then convert to int if it's an integer value
        if value[0] == 'choice':
            hp_space[key] = [value[0]] + [int(v) if v.is_integer() else v for v in hp_space[key][1:]]
    hyperopt_config['hp_space'] = hp_space
    return hyperopt_config

def get_data(dataloaders):
    train_features, train_target = [], []
    test_features, test_target = [], []

    if len(dataloaders) > 0:
        for inputs, targets in dataloaders[0]:
            train_features.append(inputs.cpu().numpy())
            train_target.append(targets.cpu().numpy())
        train_features = np.vstack(train_features)
        train_target = np.concatenate(train_target)

    if len(dataloaders) > 2:
        for inputs, targets in dataloaders[2]:
            test_features.append(inputs.cpu().numpy())
            test_target.append(targets.cpu().numpy())
        test_features = np.vstack(test_features)
        test_target = np.concatenate(test_target)

    return train_features, test_features, train_target, test_target

def objective(dataset, config, *, input_scaling, N, sr, lr, ridge, seed):
    x_train, x_test, y_train, y_test = dataset
    instances = config["instances_per_trial"]
    variable_seed = int(seed)  # Ensure seed is an integer

    losses = []
    r2s = []
    for n in range(instances):
        reservoir = Reservoir(units=int(N), sr=sr, lr=lr, input_scaling=input_scaling, seed=variable_seed)
        readout = Ridge(ridge=ridge)
        model = reservoir >> readout
        model = model.fit(x_train, y_train)
        predictions = model.run(x_test)
        loss = nrmse(y_test, predictions, norm_value=np.ptp(x_train))
        r2 = rsquare(y_test, predictions)
        variable_seed += 1
        losses.append(loss)
        r2s.append(r2)
    return {'loss': np.mean(losses), 'r2': np.mean(r2s)}

def main(gridsearch_path):
    # Read the gridsearch.yaml
    print(f"Loading configuration from {gridsearch_path}")
    config = read_yaml(gridsearch_path)
    
    # Extract the hyperopt configuration
    hyperopt_config = process_hyperopt_config(config['hyperopt_config'])
    
    results_base_dir = config['results_base_dir']
    os.makedirs(results_base_dir, exist_ok=True)

    # Save the hyperopt configuration to a JSON file
    config_path = os.path.join(results_base_dir, f"{hyperopt_config['exp']}.config.json")
    with open(config_path, "w+") as f:
        json.dump(hyperopt_config, f)

    # Get the data
    dataloaders, amplification, info_dict = get_dataloaders(config)
    dataset = get_data(dataloaders)
    
    # Hyperparameter optimization
    best_params = research(objective, dataset, config_path, ".")
    
    print(f'Best hyperparameters: {best_params}')

    # Plot hyperopt report
    fig = plot_hyperopt_report(hyperopt_config["exp"], ("lr", "sr", "ridge"), metric="r2")
    plt.savefig(os.path.join(results_base_dir, 'HyperParameterESN.png'))

if __name__ == "__main__":
    main(r"C:\Users\User\Documents\brains\brains\configs\gridsearch\gridsearch_esn.yaml")
