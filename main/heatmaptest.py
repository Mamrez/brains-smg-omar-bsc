import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import itertools

# Custom read_yaml function
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict

# Function to generate heatmap from results
def generate_heatmap(results_df, x_param, y_param, value_param, output_path, fixed_params):
    print("Generating heatmap...")
    print(f"Columns in results DataFrame: {results_df.columns.tolist()}")

    # Filter the DataFrame to only include the relevant fixed parameters
    for param, value in fixed_params.items():
        results_df = results_df[results_df[param] == value]

    # Remove prefixes for better readability
    x_label = x_param.split('.')[-1]
    y_label = y_param.split('.')[-1]

    heatmap_data = results_df.pivot(index=y_param, columns=x_param, values=value_param)
    sns.heatmap(heatmap_data, annot=True, fmt=".4f")
    plt.title(f"Heatmap of {value_param} based on {x_label} and {y_label}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(output_path)
    plt.close()

# Main function to generate heatmaps from CSV and config files
def generate_heatmaps_from_csv(csv_path, config_dir, output_dir):
    # Read the results CSV file
    results_df = pd.read_csv(csv_path)

    # Extract the list of hyperparameters from the CSV columns
    param_names = [col for col in results_df.columns if col not in ['config', 'train_loss', 'val_loss']]

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate heatmaps for all combinations of tunable hyperparameters
    fixed_params = {}
    for config_file in os.listdir(config_dir):
        if config_file.endswith('.yaml'):
            config_path = os.path.join(config_dir, config_file)
            base_config = read_yaml(config_path)
            for param in param_names:
                keys = param.split('.')
                sub_config = base_config
                try:
                    for key in keys:
                        sub_config = sub_config[key]
                    fixed_params[param] = sub_config
                except KeyError:
                    continue  # Skip if the parameter is not present in the base config

    for param_combination in itertools.combinations(param_names, 2):
        x_param, y_param = param_combination
        heatmap_output_path = os.path.join(output_dir, f'heatmap_{x_param.split(".")[-1]}_vs_{y_param.split(".")[-1]}.png')
        generate_heatmap(results_df, x_param, y_param, 'val_loss', heatmap_output_path, fixed_params)

    # Print out the ID of the lowest validation loss model
    best_model_id = results_df.loc[results_df['val_loss'].idxmin()]['config']
    print(f"The model with the lowest validation loss is: {best_model_id}")

if __name__ == "__main__":
    # Example usage
    csv_path = "main/mainTrainingData_HyperTuning/LSTM/results.csv"
    config_dir = "configs/training/Tuning/LSTM"
    output_dir = "main/mainTrainingData_HyperTuning/LSTM"
    generate_heatmaps_from_csv(csv_path, config_dir, output_dir)
