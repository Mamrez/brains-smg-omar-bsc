import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bspysmg.data import sampling
from bspysmg.data.postprocess import post_process
from brainspy.utils.io import load_configs
CONFIGS = load_configs('configs\sampling\sampling_configs_template_cdaq_to_cdaq_omar.yaml')


sampler = sampling.Sampler(CONFIGS)

sampler.sample()
sampler.close_driver()