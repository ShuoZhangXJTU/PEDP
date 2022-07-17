from hyperopt import hp
import socket
from ray import tune
from args import get_parser
from best_hyperpara import best_hyperpara
import numpy as np


def get_space(model):
    combined_space = get_parser()
    combined_space = vars(combined_space)
    for key_, val_ in best_hyperpara['pedpcls'].items():
        combined_space[key_] = val_
    combined_space['gumbel'] = True
    # -------- BASIC ----------------
    base_space = {
        # 'paths': tune.choice([1, 2, 3, 4, 5, 6]),

        # 'pred_gamma': hp.loguniform("pred_gamma", np.log(1e-3), np.log(1e+3)),
        # 'plan_gamma': hp.loguniform("plan_gamma", np.log(1e-3), np.log(1e+3)),
        # 'term_gamma': hp.loguniform("term_gamma", np.log(1e-3), np.log(1e+3)),
        # 'state_gamma': hp.loguniform("state_gamma", np.log(1e-3), np.log(1e+3)),
        # 'plan_steps': tune.choice([1, 3, 5]),
        # 'aggr_type': tune.choice(['none', 'avg', 'attn', 'vote']),
        # 'tau_plan_a': tune.choice([1e-3, 1e-2]),
        # 'tau_plan_t': tune.choice([1e-3, 1e-2]),
        # 'full_data': tune.choice([True, False]),
        # 'multilayer': tune.choice([True, False]),
        'residual': tune.choice([True, False]),
        # 'worldtf': tune.choice([True, False]),
        # 'h_dim': tune.choice([200, 512]),
        # 'dropout': tune.choice([0., 0.2]),
        'lr': tune.uniform(1e-4, 1e-3),
        'beta_decay': tune.choice([9, 18, 36]),
        'pred_gamma': tune.loguniform(1e-2, 1e+2, 1e-2),
        'plan_gamma': tune.loguniform(1e-2, 1e+2, 1e-2),
        'state_gamma': tune.loguniform(1e-2, 1e+2, 1e-2),
        'term_gamma': tune.loguniform(1e-2, 1e+2, 1e-2),
        # 'beta_upper': tune.choice([1., 0.9, 0.5, 0.25]),
        # 'beta_lower': tune.choice([0.25, 0., 0.5]),
        # 'beta_decay': tune.choice([0.05, 0.]),

        # 'LN': tune.choice([True, False]),
        # 'temperature': tune.choice([1e-4, 1e-3, 1e-2]),
        # 'h_dim': tune.choice([128, 256, 512]),

    }

    # base_space = {
    #     'tau_plan_a': tune.grid_search([1e-1, 1e-2, 1e-3, 1e-4, 1e-5]),
    #     'tau_plan_t': tune.grid_search([1e-1, 1e-2, 1e-3, 1e-4, 1e-5]),
    #     'temperature': tune.grid_search([1e-1, 1e-2, 1e-3, 1e-4, 1e-5]),
    # }

    combined_space.update(base_space)

    current_best_params = {
        # 'multilayer': False,
        'residual': False,
        # 'worldtf': False,
        # "h_dim": 512,
        # "dropout": 0.2,
        "beta_decay": 9,
        # "beta_upper": 1,
        # "beta_lower": 0.25,
        "plan_gamma": 0.09,
        "pred_gamma": 3,
        "state_gamma": 0.03,
        "term_gamma": 0.01,
        "lr": 1e-3,
    }

    return combined_space, [current_best_params], ['config/'+x for x in base_space.keys()]


def get_resources():
    server_ip = get_host_ip()
    if server_ip == '219.245.186.41':
        basic_server = Server('41', cpu=48, gpu=4, mem=32, num_samples=1)
    elif server_ip == '219.245.186.43':
        basic_server = Server('43', cpu=20, gpu=2, mem=16, num_samples=36)
    elif server_ip == '219.245.186.45':
        basic_server = Server('45', cpu=72, gpu=0, mem=32, num_samples=360)
    elif server_ip == '219.245.186.55':
        basic_server = Server('55', cpu=144, gpu=0, mem=1400, num_samples=1)
    elif server_ip == '219.245.185.191':
        basic_server = Server('191', cpu=72, gpu=0, mem=1400, num_samples=1)
    elif server_ip == '219.245.185.192':
        basic_server = Server('192', cpu=72, gpu=0, mem=1400, num_samples=1)
    elif server_ip == '219.245.185.194':
        basic_server = Server('194', cpu=72, gpu=0, mem=1400, num_samples=1)
    else:
        raise Exception("[Unseen Server]. '{}' isn't concluded in function 'get_resources'".format(server_ip))

    return basic_server


class Server:
    def __init__(self, name=None, cpu=0, gpu=0, mem=0, num_samples=1):
        self.name = name
        self.cpu = cpu
        self.gpu = gpu
        self.gpu_mem = mem
        self.num_samples = num_samples
        self.gpu_per_trial = 0.5
        if name == '41':
            self.cpu_per_trial = 1
            self.gpu_per_trial = 0
        if name == '43':
            self.gpu_per_trial = 1
            self.cpu_per_trial = 2
        if name == '45':
            self.gpu_per_trial = 0.2
            self.cpu_per_trial = 0
        if name == '55':
            self.gpu_per_trial = 0
            self.cpu_per_trial = 1


def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(('10.0.0.1', 8080))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip