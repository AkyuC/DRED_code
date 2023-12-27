from config.config import get_config as init
from env.env import env as myenv
from model.PPO import PPOCHSeletion as PPO
from utils.log_utils import get_model_path, get_object_path

import numpy as np
import random
from copy import deepcopy
from statistics import mean
import torch


def AlgGreedy_With_Minimize_Sum_Energy_Consume(seed=0, flag_save_total_energy=True):
    config = init()

    np.random.seed(seed)
    random.seed(seed)

    env = myenv(config)
    env_tmp = deepcopy(env)
    done = False
    while not done:
        energy_all_nodes = np.array(env.get_node_energy())

        if flag_save_total_energy:
            with open("energy_data/Greedy_total_energy",'a+') as f:
                f.write(str(np.sum(energy_all_nodes)) + "\n")
        if flag_save_total_energy:
            with open("energy_data/Greedy_node_energy_range",'a+') as f:
                f.write(str(mean(energy_all_nodes)) + " " + 
                        str(max(energy_all_nodes)) + " " + 
                        str(min(energy_all_nodes)) + " " + "\n")
                
        min_energy_consume = 10000
        min_idx = -1
        for idx in range(env.num_node):
            for node_idx in range(env.num_node):
                env_tmp.node[node_idx].energy = env.node[node_idx].energy
            _,_ = env_tmp.interval_step(idx)
            energy_next = np.array(env_tmp.get_node_energy())
            if min_energy_consume > sum(energy_all_nodes - energy_next):
                min_energy_consume = sum(energy_all_nodes - energy_next)
                min_idx = idx

        _, done = env.interval_step(min_idx)

        if flag_save_total_energy:
            with open("energy_data/Greedy_total_energy_consumed",'a+') as f:
                f.write(str(np.sum(energy_all_nodes)-np.sum(np.array(env.get_node_energy()))) + "\n")

    print(f"AlgGreedy_With_Minimize_Sum_Energy_Consume: {env.cnt_transmit}")
    print(env.get_node_energy())
    return env.cnt_transmit


def AlgRandom(seed=0, flag_save_total_energy=True):
    config = init()

    np.random.seed(seed)
    random.seed(seed)

    env = myenv(config)
    done = False
    while not done:
        energy_all_nodes = np.array(env.get_node_energy())

        if flag_save_total_energy:
            with open("energy_data/Random_total_energy",'a+') as f:
                f.write(str(np.sum(energy_all_nodes)) + "\n")
        if flag_save_total_energy:
            with open("energy_data/Random_node_energy_range",'a+') as f:
                f.write(str(mean(energy_all_nodes)) + " " + 
                        str(max(energy_all_nodes)) + " " + 
                        str(min(energy_all_nodes)) + " " + "\n")

        center_node = random.randint(0,19)
        _, done = env.interval_step(center_node)

        if flag_save_total_energy:
            with open("energy_data/Random_total_energy_consumed",'a+') as f:
                f.write(str(np.sum(energy_all_nodes)-np.sum(np.array(env.get_node_energy()))) + "\n")

    print(f"Random: {env.cnt_transmit}")
    print(env.get_node_energy())
    
    return env.cnt_transmit

def AlgMaxEnergy(seed=0, flag_save_total_energy=True):
    config = init()

    np.random.seed(seed)
    random.seed(seed)

    env = myenv(config)
    done = False
    while not done:
        energy_all_nodes = env.get_node_energy()
        center_node = energy_all_nodes.index(max(energy_all_nodes))
        energy_all_nodes = np.array(energy_all_nodes)

        if flag_save_total_energy:
            with open("energy_data/MaxEnergy_total_energy",'a+') as f:
                f.write(str(np.sum(energy_all_nodes)) + "\n")
        if flag_save_total_energy:
            with open("energy_data/MaxEnergy_node_energy_range",'a+') as f:
                f.write(str(mean(energy_all_nodes)) + " " + 
                        str(max(energy_all_nodes)) + " " + 
                        str(min(energy_all_nodes)) + " " + "\n")

        _, done = env.interval_step(center_node)

        if flag_save_total_energy:
            with open("energy_data/MaxEnergy_total_energy_consumed",'a+') as f:
                f.write(str(np.sum(energy_all_nodes)-np.sum(np.array(env.get_node_energy()))) + "\n")

    print(f"MaxEnergy: {env.cnt_transmit}")
    print(env.get_node_energy())
    
    return env.cnt_transmit

def AlgStatic(node=12, seed=1, flag_save_total_energy=True):
    config = init()

    np.random.seed(seed)
    random.seed(seed)

    env = myenv(config)
    done = False
    while not done:
        energy_all_nodes = np.array(env.get_node_energy())

        if flag_save_total_energy:
            with open(f"energy_data/Static_n{node}_total_energy",'a+') as f:
                f.write(str(np.sum(energy_all_nodes)) + "\n")
        if flag_save_total_energy:
            with open(f"energy_data/Static_n{node}_energy_range",'a+') as f:
                f.write(str(mean(energy_all_nodes)) + " " + 
                        str(max(energy_all_nodes)) + " " + 
                        str(min(energy_all_nodes)) + " " + "\n")

        _, done = env.interval_step(node-1)

        if flag_save_total_energy:
            with open(f"energy_data/Static_n{node}_total_energy_consumed",'a+') as f:
                f.write(str(np.sum(energy_all_nodes)-np.sum(np.array(env.get_node_energy()))) + "\n")

    print(f"Static Node{node}: {env.cnt_transmit}")
    return env.cnt_transmit

def DRED_test_best(config, seed_set=[5], flag_save_total_energy=True):
    config = config
    env = myenv(config)
    model = PPO(config)
    CHCount = {}
    for i in range(env.num_node):
        CHCount[i+1] = 0
    for seed in seed_set:
        config['seed'] = seed
        path = get_model_path(config, True)
        with open(path + '/best_episode', 'r') as file:
            episode = int(file.readline())
            model.actor_net.load_state_dict(torch.load(path + f'/actor_best.ckpt'))
            model.critic_net.load_state_dict(torch.load(path + f'/critic_best.ckpt'))

        s = config["seed"]
        np.random.seed(s)
        random.seed(s)
        torch.manual_seed(s)
        env.reset()
        done = None
        state = env.get_obs()
        while (not done) and (env.cnt_transmit < config['max_step']):
            energy_all_nodes = np.array(env.get_node_energy())

            if flag_save_total_energy:
                with open("energy_data/DRED_total_energy",'a+') as f:
                    f.write(str(np.sum(energy_all_nodes)) + "\n")
            if flag_save_total_energy:
                with open("energy_data/DRED_node_energy_range",'a+') as f:
                    f.write(str(mean(energy_all_nodes)) + " " + 
                            str(max(energy_all_nodes)) + " " + 
                            str(min(energy_all_nodes)) + " " + "\n")
                    
            for node in range(env.action_dim):
                with open(f"energy_data/DRED_node{node}_energy",'a+') as f:
                    f.write(str(energy_all_nodes[node]) + "\n")
                
            action, action_prob, probs_entropy = model.choose_abstract_action(state)
            print(action)
            CHCount[action+1]+=1
            reward, done = env.interval_step(action)
            state_next = env.get_obs() 
            state = state_next
            if done: break

            if flag_save_total_energy:
                with open("energy_data/DRED_total_energy_consumed",'a+') as f:
                    f.write(str(np.sum(energy_all_nodes)-np.sum(np.array(env.get_node_energy()))) + "\n")
    print(CHCount)
    print(env.get_node_energy())
    return env.cnt_transmit

if __name__ == '__main__':
    config = init()
    config['comm_radius'] = 100
    config['ebrp_estimate_radius'] = 100
    config['ver'] = '1.1'
    config['actor_lr'] = 1e-5
    config['critic_lr'] = 1e-4
    config['batch_size'] = 32
    config['ebrp_alpha'] = 0.1
    config['ebrp_beta'] = 0.8
    # d = {}
    # for i in range(20):
    #     d[i] = np.sqrt(np.power(config["pos_node"][i][0], 2) + np.power(config["pos_node"][i][1], 2))
    # d = sorted(d.items(), key = lambda kv:(kv[1], kv[0])) 
    # for item in d:
    #     print(item[0]+1, ',', end="")   
    DRED_test_best(config, [5], False)
    # AlgGreedy_With_Minimize_Sum_Energy_Consume()
    # AlgRandom()
    # AlgMaxEnergy()
    # AlgStatic(node=12)
