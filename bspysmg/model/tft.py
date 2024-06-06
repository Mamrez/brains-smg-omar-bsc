import pytorch_forecasting
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from torch.utils.data import DataLoader

class TFTModel:
    def __init__(self, model_structure: dict, train_data, val_data):
        self.build_model_structure(model_structure)
        self.train_data = train_data
        self.val_data = val_data

    def build_model_structure(self, model_structure: dict):
        self.model_structure = model_structure
        self.hidden_size = model_structure.get("hidden_size", 16)
        self.lstm_layers = model_structure.get("lstm_layers", 1)
        self.dropout = model_structure.get("dropout", 0.1)
        self.attention_head_size = model_structure.get("attention_head_size", 4)

    def create_datasets(self):
        training = TimeSeriesDataSet(
            self.train_data,
            time_idx="time_idx",
            target="value",
            group_ids=["group"],
            max_encoder_length=self.model_structure["max_encoder_length"],
            max_prediction_length=self.model_structure["max_prediction_length"],
            static_categoricals=["static_feature"],
            time_varying_known_reals=["time_idx", "known_feature"],
            time_varying_unknown_reals=["value"],
        )

        validation = TimeSeriesDataSet.from_dataset(training, self.val_data)

        self.train_dataloader = DataLoader(training, batch_size=32, shuffle=True)
        self.val_dataloader = DataLoader(validation, batch_size=32, shuffle=False)

    def build_model(self):
        self.model = TemporalFusionTransformer.from_dataset(
            self.train_data,
            learning_rate=0.03,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            output_size=7,  # 7 quantiles by default
            loss=pytorch_forecasting.metrics.QuantileLoss(),
        )

    def train_model(self, max_epochs=30, gpus=1):
        from pytorch_lightning import Trainer

        self.trainer = Trainer(max_epochs=max_epochs, gpus=gpus)
        self.trainer.fit(self.model, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader)
