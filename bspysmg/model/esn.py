import warnings
import numpy as np
import pandas as pd
from reservoirpy.nodes import Reservoir, Ridge

class ESNModel:
    def __init__(self, model_structure: dict, dataloaders=None):
        """
        Initialize the ESN model using the model_structure dictionary and start the training process.
        
        Parameters
        ----------
        model_structure : dict
            Dictionary containing the model structure.
            1. input_features : int
               Number of input features.
            
            2. units : int
               Number of units in the reservoir.
            
            3. lr : float
               Leakage rate of the reservoir.
            
            4. sr : float
               Spectral radius of the reservoir.
            
            5. ridge : float
               Regularization parameter for the Ridge regression.
            
            6. input_scaling : float
               Scaling applied to the input data.
            
            7. seed : int
               Seed for random number generation.
        
        dataloaders : list of DataLoader, optional
            List of dataloaders for training, validation, and test sets.
        """
        self.build_model_structure(model_structure)
        self.dataloaders = dataloaders
        if dataloaders is not None:
            self.train_model()

    def build_model_structure(self, model_structure: dict):
        """
        Build the model from the structure dictionary and set up the layers.
        
        Parameters
        ----------
        model_structure : dict
            Dictionary containing the model structure with the following keys:
            1. input_features : int
               Number of input features.
            
            2. units : int
               Number of units in the reservoir.
            
            3. lr : float
               Leakage rate of the reservoir.
            
            4. sr : float
               Spectral radius of the reservoir.
            
            5. ridge : float
               Regularization parameter for the Ridge regression.
            
            6. input_scaling : float
               Scaling applied to the input data.
            
            7. seed : int
               Seed for random number generation.
        """
        if model_structure is None:
            model_structure = {}
        self.structure_consistency_check(model_structure)
        
        self.input_features = model_structure["input_features"]
        self.units = model_structure["units"]
        self.lr = model_structure["learning_rate"]
        self.sr = model_structure["spectral_radius"]
        self.ridge = model_structure["ridge"]
        self.input_scaling = model_structure["input_scaling"]
        self.seed = model_structure["seed"]

        self.reservoir = Reservoir(units=self.units, lr=self.lr, sr=self.sr, input_scaling=self.input_scaling, seed=self.seed)
        self.readout = Ridge(ridge=self.ridge)

        # Build the ESN model
        self.model = self.reservoir >> self.readout

    def fit(self, X_train, y_train, warmup=100):
        """
        Train the ESN model.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training input data of shape (n_samples, input_features).
        
        y_train : np.ndarray
            Training output data of shape (n_samples, output_size).
        
        warmup : int
            Number of initial samples to discard for warmup.
        """
        self.model = self.model.fit(X_train, y_train, warmup=warmup)

    def predict(self, X):
        """
        Make predictions with the ESN model.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, input_features).
        
        Returns
        -------
        np.ndarray
            Predicted output data of shape (n_samples, output_size).
        """
        return self.model.run(X)

    def dataloaders_to_numpy(self):
        """
        Convert dataloaders to numpy arrays for ESN model.
        
        Returns
        -------
        tuple
            A tuple containing lists of DataFrames, inputs, and targets for each dataset (train, val, test).
        """
        if self.dataloaders is None:
            raise ValueError("Dataloaders not provided")

        datasets = []
        activation_electrode_no = self.input_features
        readout_electrode_no = self.input_features = 1

        for dataloader in self.dataloaders:
            if dataloader is not None:
                inputs = []
                targets = []
                
                for sample, target in dataloader:
                    inputs.append(sample.cpu().numpy())
                    targets.append(target.cpu().numpy())
                
                # Reshape to 2D
                inputs = np.concatenate(inputs, axis=0)
                targets = np.concatenate(targets, axis=0)
                
                df_inputs = pd.DataFrame(inputs, columns=[f'input{i}' for i in range(activation_electrode_no)])
                df_outputs = pd.DataFrame(targets, columns=[f'output{i}' for i in range(readout_electrode_no)])
                
                df = pd.concat([df_inputs, df_outputs], axis=1)
                df['time_idx'] = np.arange(len(df))
                df['group_id'] = 0
                
                datasets.append((df, inputs, targets))

        return datasets

    def train_model(self, warmup=100):
        """
        Train the ESN model using the provided dataloaders.
        
        Parameters
        ----------
        warmup : int, optional
            Number of initial samples to discard for warmup (default is 100).
        """
        datasets = self.dataloaders_to_numpy()
        # Assuming datasets[0] is train, datasets[1] is validation, and datasets[2] is test
        _, X_train, y_train = datasets[0]
        self.fit(X_train, y_train, warmup)

    def structure_consistency_check(self, model_structure: dict):
        """
        Check if the model structure follows the expected standards and set defaults if not.
        
        Parameters
        ----------
        model_structure : dict
            Dictionary of the model structure.
        
        Raises
        ------
        UserWarning
            If a parameter is not in the expected format.
        """
        default_input_features = 1
        default_units = 500
        default_lr = 0.38264094967620665
        default_sr = 0.07966435869333038
        default_ridge = 0.0029801797509298408
        default_input_scaling = 1.0
        default_seed = 1234

        if "input_features" not in model_structure:
            model_structure["input_features"] = default_input_features
            warnings.warn(
                "The model loaded does not define the input features as expected. Changed it to default value: {}.".format(default_input_features)
            )
        else:
            input_features = model_structure.get('input_features')
            assert isinstance(input_features, int) and input_features > 0, "input_features must be a positive integer"

        if "units" not in model_structure:
            model_structure["units"] = default_units
            warnings.warn(
                "The model loaded does not define the number of units as expected. Changed it to default value: {}.".format(default_units)
            )
        else:
            units = model_structure.get('units')
            assert isinstance(units, int) and units > 0, "units must be a positive integer"

        if "learning_rate" not in model_structure:
            model_structure["learning_rate"] = default_lr
            warnings.warn(
                "The model loaded does not define the leakage rate as expected. Changed it to default value: {}.".format(default_lr)
            )
        else:
            lr = model_structure.get('learning_rate')
            assert isinstance(lr, float) and lr > 0, "lr must be a positive float"

        if "spectral_radius" not in model_structure:
            model_structure["spectral_radius"] = default_sr
            warnings.warn(
                "The model loaded does not define the spectral radius as expected. Changed it to default value: {}.".format(default_sr)
            )
        else:
            sr = model_structure.get('spectral_radius')
            assert isinstance(sr, float) and sr > 0, "sr must be a positive float"

        if "ridge" not in model_structure:
            model_structure["ridge"] = default_ridge
            warnings.warn(
                "The model loaded does not define the ridge parameter as expected. Changed it to default value: {}.".format(default_ridge)
            )
        else:
            ridge = model_structure.get('ridge')
            assert isinstance(ridge, float) and ridge > 0, "ridge must be a positive float"

        if "input_scaling" not in model_structure:
            model_structure["input_scaling"] = default_input_scaling
            warnings.warn(
                "The model loaded does not define the input scaling as expected. Changed it to default value: {}.".format(default_input_scaling)
            )
        else:
            input_scaling = model_structure.get('input_scaling')
            assert isinstance(input_scaling, float) and input_scaling > 0, "input_scaling must be a positive float"

        if "seed" not in model_structure:
            model_structure["seed"] = default_seed
            warnings.warn(
                "The model loaded does not define the seed as expected. Changed it to default value: {}.".format(default_seed)
            )
        else:
            seed = model_structure.get('seed')
            assert isinstance(seed, int) and seed > 0, "seed must be a positive integer"
