import os
import torch


class model():
    def __init__(self, config, data, test=False):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.training_opt = self.config['training_opt']
        self.memory = self.config['memory']
        self.data = data
        self.test_mode = test
        self.init_models()

    def init_models(self, optimizer=True):
        networks_defs = self.config['networks']
        self.networks = {}
        self.model_optim_params_list = []

        print("Using {} GPUs".format(torch.cuda.device_count()))
        for key, val in networks_defs.items():
            

        