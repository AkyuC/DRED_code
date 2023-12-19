from utils.alg_utils import init
from utils.log_utils import get_model_path, get_object_path, mkdir
from env.env import env as myenv
from model.PPO import PPOCHSeletion as PPO
from runner.PPO_runner import PPORunner 
import torch
import numpy as np
import random
from statistics import mean
import sys


def test_episode(config, episode_list, seed_set=[1,2,3,4,5]):
    config = config
    env = myenv(config)
    model = PPO(config)
    for s_i, seed in enumerate(seed_set):
        config['seed'] = seed
        model.load(episode_list[s_i])
        result_list = []
        for s in range(1):
            np.random.seed(s)
            random.seed(s)
            torch.manual_seed(s)
            env.reset()
            done = None
            state = env.get_obs()
            # print(env.pos_hard_code)
            count = 1
            while (not done) and (env.cnt_transmit < config['max_step']):
                action, action_prob, probs_entropy = model.choose_abstract_action(state)
                # if count == 100 or count == 200:
                #     print(f"count: {count}, action: {action}")
                #     print(f"energy: {np.array(env.get_node_energy())/config['sensor_energy']*100}")
                reward, done = env.interval_step(action, count)
                # print(reward)
                state_next = env.get_obs() 
                state = state_next
                count += 1
                if done: break
            result_list.append(env.cnt_transmit)
        print(f"seed:{seed}, episode {episode_list[s_i]}, mean: {mean(result_list)}, max: {max(result_list)}, min: {min(result_list)}")


def test_best(config, seed_set=[1,2,3,4,5]):
    config = config
    env = myenv(config)
    model = PPO(config)
    for seed in seed_set:
        config['seed'] = seed
        path = get_model_path(config, True)
        with open(path + '/best_episode', 'r') as file:
            episode = int(file.readline())
            model.actor_net.load_state_dict(torch.load(path + f'/actor_best.ckpt'))
            model.critic_net.load_state_dict(torch.load(path + f'/critic_best.ckpt'))
        result_list = []
        for s in range(1):
            np.random.seed(s)
            random.seed(s)
            torch.manual_seed(s)
            env.reset()
            done = None
            state = env.get_obs()
            while (not done) and (env.cnt_transmit < config['max_step']):
                # if s == 4:
                #     with open("Ours_total_energy",'a+') as f:
                #         f.write(str(np.sum(env.get_node_energy())) + "\n")
                action, action_prob, probs_entropy = model.choose_abstract_action(state)
                reward, done = env.interval_step(action)
                print(action)
                state_next = env.get_obs() 
                state = state_next
                if done: break
            result_list.append(env.cnt_transmit)
            print(env.cnt_transmit)
        print(f"seed:{seed}, episode {episode}, mean: {mean(result_list)}, max: {max(result_list)}, min: {min(result_list)}")

def test_all(config, episode_range=[0,500]):
    config = config
    env = myenv(config)
    model = PPO(config)
    path = get_model_path(config, True)
    for episode in range(episode_range[0], episode_range[1]):
        model.actor_net.load_state_dict(torch.load(path + f'/actor{episode*10}.ckpt'))
        model.critic_net.load_state_dict(torch.load(path + f'/critic{episode*10}.ckpt'))

        result_list = []
        for s in range(10):
            np.random.seed(s)
            random.seed(s)
            torch.manual_seed(s)
            env.reset()
            done = None
            state = env.get_obs()
            while (not done) and (env.cnt_transmit < config['max_step']):
                action, action_prob, probs_entropy = model.choose_abstract_action(state)
                reward, done = env.interval_step(action)
                # print(action)
                state_next = env.get_obs() 
                state = state_next
                if done: break
            result_list.append(env.cnt_transmit)

        with open(get_object_path(config, True) + '/lifetime_max', 'a+') as f:
            f.write(str(max(result_list)) + "\n")
        with open(get_object_path(config, True) + '/lifetime_min', 'a+') as f:
            f.write(str(min(result_list)) + "\n")
        with open(get_object_path(config, True) + '/lifetime_avg', 'a+') as f:
            f.write(str(mean(result_list)) + "\n")

        print(f"episode {episode}, mean: {mean(result_list)}, max: {max(result_list)}, min: {min(result_list)}")

def get_total_energy_curve(config, seed, episode=-1):
    config = config
    env = myenv(config)
    model = PPO(config)
    config['seed'] = seed
    path = get_model_path(config, True)
    if episode==-1:
        with open(path + '/best_episode', 'r') as file:
            episode = int(file.readline())
            model.actor_net.load_state_dict(torch.load(path + f'/actor_best.ckpt'))
            model.critic_net.load_state_dict(torch.load(path + f'/critic_best.ckpt'))
    else:
        model.actor_net.load_state_dict(torch.load(path + f'/actor{episode*10}.ckpt'))
        model.critic_net.load_state_dict(torch.load(path + f'/critic{episode*10}.ckpt'))
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    env.reset()
    done = None
    state = env.get_obs()
    while (not done) and (env.cnt_transmit < config['max_step']):
        with open("Ours_total_energy",'a+') as f:
            f.write(str(np.sum(env.get_node_energy())) + "\n")
        last_en = np.sum(env.get_node_energy())
        action, action_prob, probs_entropy = model.choose_abstract_action(state)
        reward, done = env.interval_step(action)
        # print(reward)
        state_next = env.get_obs() 
        state = state_next
        # print("consumed:" + str(last_en - np.sum(env.get_node_energy())))
        if done: break
    print(env.cnt_transmit)

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
    # test_best(config, [2])
    # test_episode(config, [24380], [3])
    # test_all(config, [300,700])
    # episode = 500
    for i in range(100):
        test_episode(config, [30610+i*10], [1])
    # for episode in range(350,500):
    #     get_total_energy_curve(config, seed=1, episode=episode)
    # get_total_energy_curve(config, seed=3, episode=24386)