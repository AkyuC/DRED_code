from turtle import color
import numpy as np
import matplotlib.pyplot as plt

from utils.log_utils import get_loss_path, get_reward_path, get_object_path


def plot_loss(config):
    path = get_loss_path(config, False) + "/aloss.png"
    plt.figure(figsize=(10,3))
    loss = np.loadtxt(get_loss_path(config, True) + "/aloss")
    plt.plot(np.arange(len(loss)), loss)
    plt.ylabel("aloss")
    plt.xlabel("update_step")
    plt.title("aloss")
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.close('all')

    path = get_loss_path(config, False) + "/closs.png"
    plt.figure(figsize=(10,3))
    loss = np.loadtxt(get_loss_path(config, True) + "/closs")
    plt.plot(np.arange(len(loss)), loss)
    plt.ylabel("closs")
    plt.xlabel("update_step")
    plt.title("closs")
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.close('all')

def plot_survival_time(config):
    lifetime_path = get_object_path(config, False) + "/lifetime.png"
    plt.figure(figsize=(10,3))
    lifetime = np.loadtxt(get_object_path(config, True) + "/lifetime")
    print(max(lifetime))
    plt.plot(np.arange(len(lifetime)), lifetime, label="DRED")
    plt.plot(np.arange(len(lifetime)), np.ones(len(lifetime))*2253, label='Greedy')
    plt.plot(np.arange(len(lifetime)), np.ones(len(lifetime))*2188, label='MaxEnergy')
    plt.plot(np.arange(len(lifetime)), np.ones(len(lifetime))*1626, label='Random')
    # plt.plot(np.arange(len(lifetime)), np.ones(len(lifetime))*235, label='Greedy')
    # plt.plot(np.arange(len(lifetime)), np.ones(len(lifetime))*2549, label='MaxEnergy')
    # plt.plot(np.arange(len(lifetime)), np.ones(len(lifetime))*2451, label='Random')
    plt.legend()
    plt.ylabel("lifetime")
    plt.xlabel("episode")
    plt.title('lifetime')
    plt.savefig(lifetime_path, bbox_inches='tight', pad_inches=0.1)
    plt.close('all')

def plot_mean_survival_time(config, seed_set, datalen=4000):
    lifetime = []
    for seed in seed_set:
        config['seed'] = seed
        lifetime.append(np.loadtxt(get_object_path(config, True) + "/lifetime")[:datalen].tolist())
    lifetime = np.array(lifetime)
    avg_lifetime = np.average(lifetime, axis=0)
    max_lifetime = np.max(lifetime, axis=0)
    min_lifetime = np.min(lifetime, axis=0)
    plt.rcParams['figure.figsize'] = (7,5)
    plt.rcParams['font.size'] = 16
    plt.fill_between(np.arange(len(avg_lifetime)), max_lifetime, min_lifetime, color='coral')
    plt.plot(np.arange(len(avg_lifetime)), avg_lifetime, color="red", label="DRED", linewidth = 2)
    plt.plot(np.arange(len(avg_lifetime)), np.ones(len(avg_lifetime))*1514.6, color='black', label="Random", alpha = 0.75, linewidth = 2)
    plt.plot(np.arange(len(avg_lifetime)), np.ones(len(avg_lifetime))*2179.7, color='green', label="Greedy", alpha = 0.75, linewidth = 2)
    plt.plot(np.arange(len(avg_lifetime)), np.ones(len(avg_lifetime))*395.8, color='b', label="Static", alpha = 0.75, linewidth = 2)
    plt.ylabel("Network Lifetime / Rounds")
    plt.xlabel("Train Episode")
    # plt.title('lifetime')
    plt.legend()
    ax = plt.gca()
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='x')
    config['seed'] = seed_set[0]
    lifetime_path = "./lifetime_avg.png"
    plt.savefig(lifetime_path, bbox_inches='tight', pad_inches=0.05, dpi=600)
    lifetime_path = "./lifetime_avg.eps"
    plt.savefig(lifetime_path, format="eps", bbox_inches='tight', pad_inches=0.05, dpi=600)
    plt.close('all')
    print(max(avg_lifetime))

def plot_ppo_ratio(config):
    path = get_loss_path(config, False) + "/ratio.png"
    plt.figure(figsize=(10,3))
    ratio = np.loadtxt(get_loss_path(config, True) + "/ratio")
    plt.plot(np.arange(len(ratio)), ratio)
    plt.ylabel("ratio")
    plt.xlabel("update_step")
    plt.title("ratio")
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.close('all')

def plot_ppo_state_value(config):
    path = get_loss_path(config, False) + "/state_value.png"
    plt.figure(figsize=(10,3))
    ratio = np.loadtxt(get_loss_path(config, True) + "/state_value")
    plt.plot(np.arange(len(ratio)), ratio)
    plt.ylabel("state_value")
    plt.xlabel("update_step")
    plt.title("state_value")
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.close('all')

def get_avg_lifetime_max(config):
    lifetime_avg = np.loadtxt(get_object_path(config, True) + "/lifetime_avg")
    max_lifetime_avg = max(lifetime_avg)
    index_max_lifetime_avg = lifetime_avg.tolist().index(max_lifetime_avg)
    return index_max_lifetime_avg, max_lifetime_avg
