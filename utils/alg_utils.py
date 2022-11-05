import argparse
import os
from config.config import get_config, show_config
import torch
import numpy as np
import random

def init():
    args = argparse.ArgumentParser()
    args.add_argument("--cudaDevice", type=int, default=0, help="the cuda device")
    args.add_argument("--seed", type=int, default=0, help="the random seed")
    args.add_argument("--comm_radius", type=int, default=100, help="the communication radius of node")
    args.add_argument("--es_radius", type=int, default=100, help="the estimate radius of energy potential")
    args.add_argument("--ver", type=str, default="1.0", help="algorthm version")
    args.add_argument("--alr", type=float, default=1e-5, help="actor learning rate")
    args.add_argument("--clr", type=float, default=1e-4, help="critic learning rate")
    args.add_argument("--batch_size", type=int, default=32, help="batch_size")
    args.add_argument("--ebrp_alpha", type=float, default=0.1, help="ebrp_alpha")
    args.add_argument("--ebrp_beta", type=float, default=0.8, help="ebrp_beta")
    args.add_argument("--load_episode", type=int, default=-1, help="load_episode")
    args = args.parse_args()

    config = get_config()
    config['gpu'] = args.cudaDevice
    config['seed'] = args.seed
    config['comm_radius'] = args.comm_radius
    config['ebrp_estimate_radius'] = args.es_radius
    config['ver'] = args.ver
    config['actor_lr'] = args.alr
    config['critic_lr'] = args.clr
    config['batch_size'] = args.batch_size
    config['ebrp_alpha'] = args.ebrp_alpha
    config['ebrp_beta'] = args.ebrp_beta
    config['load_episode'] = args.load_episode

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu'])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # show_config(config)

    return config