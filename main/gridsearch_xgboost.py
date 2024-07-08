import sys
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from bspysmg.data.dataset import get_dataloaders
import matplotlib.pyplot as plt
import seaborn as sns

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict

def save_yaml(config, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(config, file)

def get_data(dataloaders):
    train_features, train_target = [], []
    val_features, val_target = [], []
    test_features, test_target = [], []

    if len(dataloaders) > 0:
        for inputs, targets in dataloaders[0]:
            train_features.append(inputs.cpu().numpy())
            train_target.append(targets.cpu().numpy())
        train_features = np.vstack(train_features)
        train_target = np.concatenate(train_target)

    if len(dataloaders) > 1:
        for inputs, targets in dataloaders[1]:
            val_features.append(inputs.cpu().numpy())
            val_target.append(targets.cpu().numpy())
        val_features = np.vstack(val_features)
        val_target = np.concatenate(val_target)

    if len(dataloaders) > 2:
        for inputs, targets in dataloaders[2]:
            test_features.append(inputs.cpu().numpy())
            test_target.append(targets.cpu().numpy())
        test_features = np.vstack(test_features)
        test_target = np.concatenate(test_target)

    return train_features, val_features, test_features, train_target, val_target, test_target

def plot_grid_search_results(csv_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Plot the scores
    plt.figure(figsize=(14, 7))
    sns.scatterplot(data=df, x='param_learning_rate', y='mean_test_score', hue='param_max_depth', style='param_colsample_bytree', size='param_n_estimators', palette='viridis', sizes=(40, 400))
    plt.title('Grid Search Scores')
    plt.xlabel('Learning Rate')
    plt.ylabel('Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(csv_path), 'grid_search_scores.png'))

    # Plot the training times
    plt.figure(figsize=(14, 7))
    sns.scatterplot(data=df, x='param_learning_rate', y='mean_fit_time', hue='param_max_depth', style='param_colsample_bytree', size='param_n_estimators', palette='viridis', sizes=(40, 400))
    plt.title('Grid Search Training Times')
    plt.xlabel('Learning Rate')
    plt.ylabel('Time (seconds)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(csv_path), 'grid_search_training_times.png'))

def plot_predictions(true_values, predictions,csv_path, title='Predicted vs Actual'):
    plt.figure(figsize=(10, 5))
    plt.plot(true_values, label='True Values')
    plt.plot(predictions, label='Predictions', alpha=0.7)
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(csv_path, 'grid_search_test_prediction.png'))

def main(gridsearch_path):
    # Read the gridsearch.yaml
    print(f"Loading configuration from {gridsearch_path}")
    config = read_yaml(gridsearch_path)
    
    # Extract the base configuration and hyperparameter ranges
    base_config = config.copy()
    param_ranges = config.pop('hyperparameter_ranges')
    results_base_dir = base_config['results_base_dir']
    config_base_dir = base_config['config_base_dir']
    os.makedirs(results_base_dir, exist_ok=True)
    os.makedirs(config_base_dir, exist_ok=True)
    
    # Convert the range dictionary values into lists for param_grid
    param_grid = {}
    for param, range_spec in param_ranges.items():
        if isinstance(range_spec, dict) and 'start' in range_spec and 'stop' in range_spec and 'step' in range_spec:
            param_grid[param.split('.')[-1]] = list(range(range_spec['start'], range_spec['stop'], range_spec['step']))
        elif isinstance(range_spec, dict) and 'values' in range_spec:
            param_grid[param.split('.')[-1]] = range_spec['values']
        else:
            raise ValueError(f"Invalid range specification for parameter: {param}")

    # Get the data
    dataloaders, amplification, info_dict = get_dataloaders(config)
    train_features, val_features, test_features, train_target, val_target, test_target = get_data(dataloaders)

    try:
        # Initialize XGBRegressor
        xgb_model = XGBRegressor(objective=info_dict['model_structure']['objective'])

        # Grid Search
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=2, scoring='neg_mean_squared_error', verbose=3)
        grid_search.fit(train_features, train_target)

        # Save grid search results
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_csv_path = os.path.join(results_base_dir, 'grid_search_results.csv')
        results_df.to_csv(results_csv_path, index=False)

        # Plot grid search results
        plot_grid_search_results(results_csv_path)

        # Best parameters
        print(f'Best parameters: {grid_search.best_params_}')

        # Train with best parameters
        best_model = grid_search.best_estimator_

        # Validate the model
        predictions = best_model.predict(val_features)
        val_loss = mean_squared_error(val_target, predictions, squared=False)

        print(f"Validation loss: {val_loss:.6f}")

        # Evaluate on test data
        predictions_best = best_model.predict(test_features)
        rmse = mean_squared_error(test_target, predictions_best, squared=False)
        print(f'RMSE on test data: {rmse}')

        # Plot predictions
        plot_predictions(test_target, predictions_best, title='Test Set: Actual vs Predicted',csv_path=results_base_dir)

    except KeyboardInterrupt:
        print("Interrupted by user")

if __name__ == "__main__":
    main("configs/gridsearch/gridsearch_xgboost.yaml")
