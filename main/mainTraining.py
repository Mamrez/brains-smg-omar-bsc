import matplotlib
matplotlib.use('Agg') # To allow non-graphical plotting
from bspysmg.model.training import generate_surrogate_model
from brainspy.utils.io import load_configs
from bspysmg.data.postprocess import post_process

inputs, outputs, info_dictionary = post_process('main/mainSamplingDataFull/', clipping_value=None)
print(f"max out {outputs.max()} max min {outputs.min()} shape {outputs.shape}")
      

smg_configs = load_configs('configs/fulltraining/smg_configs_template_omar.yaml')
generate_surrogate_model(smg_configs)