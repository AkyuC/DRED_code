from copy import deepcopy
from enum import Flag
from math import sqrt
from statistics import mean
from time import time
from config.config import get_config as init
from env.env import env as myenv

import numpy as np
import random


def AlgGreedy(seed=0):
    config = init()

    np.random.seed(seed)
    random.seed(seed)

    env = myenv(config)
    done = False
    while not done:
        energy_all_nodes = env.get_node_energy()
        _, done = env.interval_step(energy_all_nodes.index(max(energy_all_nodes)))

    print(f"Greedy: {env.cnt_transmit}")
    print(env.get_node_energy())
    # print(env.send_consomed_en)
    # print(env.recv_consomed_en)
    return env.cnt_transmit

def AlgGreedy_With_Minimize_Sum_Energy_Consume(seed=0, flag_save_total_energy=False):
    config = init()

    np.random.seed(seed)
    random.seed(seed)

    env = myenv(config)
    env_tmp = deepcopy(env)
    done = False
    while not done:
        energy_all_nodes = np.array(env.get_node_energy())
        if flag_save_total_energy:
            with open("Greedy_total_energy",'a+') as f:
                f.write(str(np.sum(energy_all_nodes)) + "\n")
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

        # print(min_idx)
        # route = env.proc_inter.get_route(env.node, min_idx)
        # route_sort = [-1 for _ in range(20)]
        # for i in range(19):
        #     route_sort[route[i][0]] = route[i][1]
        # print(route_sort)
        _, done = env.interval_step(min_idx)

    print(f"AlgGreedy_With_Minimize_Sum_Energy_Consume: {env.cnt_transmit}")
    print(env.get_node_energy())
    return env.cnt_transmit

def AlgRandom(seed=0, flag_save_total_energy=False):
    config = init()

    np.random.seed(seed)
    random.seed(seed)

    env = myenv(config)
    done = False
    avg_time = 0
    a = []
    while not done:
        energy_all_nodes = np.array(env.get_node_energy())
        if flag_save_total_energy:
            with open("Random_total_energy",'a+') as f:
                f.write(str(np.sum(energy_all_nodes)) + "\n")
        t1 = time()
        center_node = random.randint(0,19)
        avg_time += time()-t1
        _, done = env.interval_step(center_node)
        a.append(sum(energy_all_nodes) - sum(env.get_node_energy()))
    print(min(a))
    print(max(a))
    print(f"Random: {env.cnt_transmit}")
    print(avg_time/int(env.cnt_transmit/10))
    print(env.get_node_energy())
    
    return env.cnt_transmit

def AlgMaxEnergy(seed=0, flag_save_total_energy=False):
    config = init()

    np.random.seed(seed)
    random.seed(seed)

    env = myenv(config)
    done = False
    avg_time = 0
    a = []
    while not done:
        energy_all_nodes = env.get_node_energy()
        center_node = energy_all_nodes.index(max(energy_all_nodes))
        energy_all_nodes = np.array(energy_all_nodes)
        if flag_save_total_energy:
            with open("MaxEnergy_total_energy",'a+') as f:
                f.write(str(np.sum(energy_all_nodes)) + "\n")
        t1 = time()
        avg_time += time()-t1
        _, done = env.interval_step(center_node)
        a.append(sum(energy_all_nodes) - sum(env.get_node_energy()))
    print(min(a))
    print(max(a))
    print(f"MaxEnergy: {env.cnt_transmit}")
    # print(avg_time/int(env.cnt_transmit/10))
    print(env.get_node_energy())
    
    return env.cnt_transmit

def AlgLeach_F(seed=0):
    config = init()

    np.random.seed(seed)
    random.seed(seed)

    env = myenv(config)
    done = False
    idx = 0
    while not done:
        _, done = env.interval_step(idx)
        idx = (idx+1)%20

    print(f"LEACH-F: {env.cnt_transmit}")
    return env.cnt_transmit

def AlgStatic(seed=0, node=-1, flag_save_total_energy=False):
    config = init()
    ret = []

    if node != -1:
        np.random.seed(seed)
        random.seed(seed)

        env = myenv(config)
        done = False
        while not done:
            energy_all_nodes = np.array(env.get_node_energy())
            if flag_save_total_energy:
                with open(f"Static_n{node}_total_energy",'a+') as f:
                    f.write(str(np.sum(energy_all_nodes)) + "\n")
            _, done = env.interval_step(node-1)

        print(f"Static Node{node}: {env.cnt_transmit}")
        ret.append(env.cnt_transmit)
    else:
        for idx in range(20):
            np.random.seed(seed)
            random.seed(seed)

            env = myenv(config)
            done = False
            while not done:
                _, done = env.interval_step(idx)

            print(f"Static Node{idx}: {env.cnt_transmit}")
            ret.append(env.cnt_transmit)
    return ret

def AlgRotate(seed=0):
    config = init()
    ret = []

    env = myenv(config)
    node_dist = []
    for node in env.node:
        node_dist.append((sqrt(node.pos[0] ** 2 + node.pos[1] ** 2), node.node_id))
    def takefirst(elem):
        return elem[0]
    node_dist.sort(key=takefirst)
    idx_list = [n[1] for n in node_dist]

    for idx_r in range(len(idx_list)):
        for seed in range(1):
            env.reset()
            np.random.seed(seed)
            random.seed(seed)
            done = False
            idx = 0
            while not done:
                _, done = env.interval_step(idx_list[idx])
                idx = (idx+1)%(idx_r+1)
            ret.append(env.cnt_transmit)

        print(f"Rotate: {env.cnt_transmit}, idx_list: {idx_list[0:idx_r+1]}")
        # print(f"Rotate,idx_list{idx_list}, mean: {mean(ret)}, max: {max(ret)}, min: {min(ret)}")
        # print(env.get_node_energy())
        # return env.cnt_transmit


if __name__ == '__main__':
    # AlgRotate()
    AlgRandom(0,False)
    t1 = time()
    # AlgGreedy_With_Minimize_Sum_Energy_Consume(0,True)
    AlgGreedy_With_Minimize_Sum_Energy_Consume(0,False)
    AlgMaxEnergy(0,False)
    # print('程序运行时间:%s毫秒' % ((time() - t1)*1000))
    # t1 = time()
    # AlgGreedy()
    # print('程序运行时间:%s毫秒' % ((time() - t1)*1000))
    # t1 = time()
    # AlgRandom(0,True)
    # print('程序运行时间:%s毫秒' % ((time() - t1)*1000))
    # t1 = time()
    # AlgLeach_F()
    # print('程序运行时间:%s毫秒' % ((time() - t1)*1000))
    # t1 = time()
    # AlgStatic()
    # print('程序运行时间:%s毫秒' % ((time() - t1)*1000))
    # AlgStatic(0,13,True)

    # result = [[],[],[],[],[]]
    # for seed in range(10):
    #     result[0].append(AlgGreedy_With_Minimize_Sum_Energy_Consume(seed))
    #     # result[1].append(AlgGreedy(seed))
    #     result[2].append(AlgRandom(seed))
    #     result[3].append(AlgMaxEnergy(seed))
    #     result[4].append(AlgStatic(seed))

    # print(" \n")
    # print(result)

    # print(" \n")
    # print(f"AlgGreedy_With_Minimize_Sum_Energy_Consume, mean: {mean(result[0])}, max: {max(result[0])}, min: {min(result[0])}")
    # # print(f"Greedy, mean: {mean(result[1])}, max: {max(result[1])}, min: {min(result[1])}")
    # print(f"Random, mean: {mean(result[2])}, max: {max(result[2])}, min: {min(result[2])}")
    # print(f"AlgMaxEnergy, mean: {mean(result[3])}, max: {max(result[3])}, min: {min(result[3])}")
    # result_static = np.array(result[4])
    # print(f"Static, mean: {result_static.mean(axis=0)}, max: {result_static.max(axis=0)}, min: {result_static.min(axis=0)}")

