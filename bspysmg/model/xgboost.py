import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error

class XGBoostModel:
    def __init__(self, model_structure: dict, train, test):
        self.train_model(model_structure, train, test)


    def dataloader_to_dmatrix(self, dataloader):
        data = []
        labels = []
        for batch in dataloader:
            inputs, targets = batch
            data.append(inputs.cpu().numpy())
            labels.append(targets.cpu().numpy())
        data = np.vstack(data)
        labels = np.concatenate(labels)
        return xgb.DMatrix(data, label=labels)

    def train_model(self, model_structure: dict, train, test):
        if model_structure is None:
            model_structure = {}
        self.structure_consistency_check(model_structure)

        self.params = {
            'objective': model_structure["objective"],
            'eval_metric': model_structure["eval_metric"],
            'colsample_bytree': model_structure["colsample_bytree"],
            'learning_rate': model_structure["learning_rate"],
            'max_depth': model_structure["max_depth"],
            'subsample': model_structure["subsample"]
        }

        self.num_boost_round = model_structure["n_estimators"]

        # Extract model parameters
        # self.objective = model_structure["objective"]
        # self.evaL_metric = model_structure["eval_metric"]
        # self.colsample_bytree = model_structure["colsample_bytree"]
        # self.learning_rate = model_structure["learning_rate"]
        # self.max_depth = model_structure["max_depth"]
        # self.n_estimators = model_structure["n_estimators"]
        # self.subsample = model_structure["subsample"]
        # self.input_features = model_structure["input_features"]
        # self.num_boost_round = model_structure["num_boost_round"]

        # Convert data into DMatrix format
        dtrain = self.dataloader_to_dmatrix(train)
        dtest = self.dataloader_to_dmatrix(test)

        self.dtest = dtest
        self.model = xgb.train(self.params, dtrain, self.num_boost_round)


    def structure_consistency_check(self, model_structure: dict):
        defaults = {
            "objective": 'reg:squarederror',
            "eval_metric": 'rmse',
            "colsample_bytree": 1.0,
            "learning_rate": 0.2,
            "max_depth": 7,
            "n_estimators": 1000,
            "subsample": 1.0,
            "input_features": 7,
            "num_boost_round": 200
        }

        if "objective" not in model_structure:
            model_structure["objective"] = defaults["objective"]
        else:
            objective = model_structure.get("objective")
            assert isinstance(objective, str), "objective must be a string"

        if "eval_metric" not in model_structure:
            model_structure["eval_metric"] = defaults["eval_metric"]
        else:
            eval_metric = model_structure.get("eval_metric")
            assert isinstance(eval_metric, str), "eval_metric must be a string"

        if "colsample_bytree" not in model_structure:
            model_structure["colsample_bytree"] = defaults["colsample_bytree"]
        else:
            colsample_bytree = model_structure.get("colsample_bytree")
            assert isinstance(colsample_bytree, float) and 0 < colsample_bytree <= 1, "colsample_bytree must be a float between 0 and 1"

        if "learning_rate" not in model_structure:
            model_structure["learning_rate"] = defaults["learning_rate"]
        else:
            learning_rate = model_structure.get("learning_rate")
            assert isinstance(learning_rate, float) and learning_rate > 0, "learning_rate must be a positive float"

        if "max_depth" not in model_structure:
            model_structure["max_depth"] = defaults["max_depth"]
        else:
            max_depth = model_structure.get("max_depth")
            assert isinstance(max_depth, int) and max_depth > 0, "max_depth must be a positive integer"

        if "n_estimators" not in model_structure:
            model_structure["n_estimators"] = defaults["n_estimators"]
        else:
            n_estimators = model_structure.get("n_estimators")
            assert isinstance(n_estimators, int) and n_estimators > 0, "n_estimators must be a positive integer"

        if "subsample" not in model_structure:
            model_structure["subsample"] = defaults["subsample"]
        else:
            subsample = model_structure.get("subsample")
            assert isinstance(subsample, float) and 0 < subsample <= 1, "subsample must be a float between 0 and 1"

        if "input_features" not in model_structure:
            model_structure["input_features"] = defaults["input_features"]
        else:
            input_features = model_structure.get("input_features")
            assert isinstance(input_features, int) and input_features > 0, "input_features must be a positive integer"

        if "num_boost_round" not in model_structure:
            model_structure["num_boost_round"] = defaults["num_boost_round"]
        else:
            num_boost_round = model_structure.get("num_boost_round")
            assert isinstance(num_boost_round, int) and num_boost_round > 0, "num_boost_round must be a positive integer"

    def predict(self, dtest):
        return self.model.predict(dtest)
