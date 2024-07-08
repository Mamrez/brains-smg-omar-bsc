import sys
import os
import yaml
import itertools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import matplotlib
matplotlib.use('Agg') # To allow non-graphical plotting
from bspysmg.model.training import generate_surrogate_model
from brainspy.utils.io import load_configs
from bspysmg.data.postprocess import post_process
from bspysmg.model.lstm import LSTMModel
from bspysmg.model.gru import GRUModel
from bspysmg.model.xgboost import XGBoostModel
from bspysmg.model.esn import ESNModel
import torch


torch.cuda.init()

#inputs, outputs, info_dictionary = post_process('main\mainSamplingData', clipping_value=None)
#print(f"max out {outputs.max()} max min {outputs.min()} shape {outputs.shape}")


smg_configs = load_configs('configs/training/smg_configs_template_omar_esn.yaml')
generate_surrogate_model(smg_configs, custom_model=ESNModel)