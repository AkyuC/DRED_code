# write / read config file with json module
import json


def init_config_file() -> None:
    config = dict({
                # the initial energy for different types nodes
                'sensor_energy': 0.15,
                'died_threshold': 1e-4,

                # the unit cost
                'unit_cost_rec': 50e-9, # /b
                'unit_cost_send': 1e-12, # /b/m^2
                'unit_cost_proc': 0,   # /b

                # communication radius
                'comm_radius': 100,

                # iot data set
                'data_max': 1000,
                'data_min': 500,

                # for RL agent, nn params
                'state_dim': 60,
                'action_dim': 20,
                'actor_lr':1e-5,
                'critic_lr':1e-4,
                'gamma': 0.99,
                'gae_lambda': 0.9, 
                'clip_param': 0.15,
                'max_grad_norm': 0.5,
                'ppo_update_time': 4,
                'buffer_capacity': 1000,
                'batch_size': 64,
                'energy_max': 0.045,
                'energy_min': 0.01,
                'entropy_coef':1e-2,
                'vf_coef': 0.5,
                'env_n': 1,
                'env_step': 128,

                # runner
                'max_step': 99999,
                'max_episode': 999999,
                'trans_in_interval': 10,

                # EBRP
                'ebrp_alpha': 0.1,
                'ebrp_beta': 0.8,
                'ebrp_estimate_radius': 100,

                # others
                'alg': 'PPO',
                'save_freq': 10,
                'map_range': 100,
                'collected_dist': 50,
                'round_baseline': 1800,
                'round_base': 100,
                'device': 'cuda',
                'gpu': 0,
                'seed': 0,
                })

    with open('./config/config.json', 'w') as f:
        json.dump(config, f)



# read config content from specified json file
def get_config(filename='./config/config'):
    init_config_file()
    f = open(filename + '.json')
    ob = json.load(f)
    f.close()
    # ob['pos_node'] = [(-298, -792), (-500, 200), (394, 598), (-900, -254), (162, -920),
    #         (-124, 340), (66, 450), (-160, 148), (600, -250), (160, -24),
    #         (-320, -600), (650, -454), (296, -442), (-590, 560), (640, 424),
    #         (-96, -130), (-696, -256), (-336, 390), (100, 640), (-152, 712)]
    ob['pos_node'] = [(-29, -79), (-50, 20), (39, 59), (-90, -25), (16, -92), 
                    (-12, 34), (6, 45), (-16, 14), (60, -25), (16, -2), 
                    (-32, -60), (65, -45), (29, -44), (-59, 56), (64, 42), 
                    (-9, -13), (-69, -25), (-33, 39), (10, 64), (-15, 71)]
    # ob['pos_node'] = [(-99, -264), (-166, 66), (131, 199), (-300, -84), (54, -306), (-41, 113), (22, 150), (-53, 49), (200, -83), (53, -8), (-106, -200), (216, -151), (98, -147), (-196, 186), (213, 141), (-32, -43), (-232, -85), (-112, 130), (33, 213), (-50, 237)]
    
    return ob

# set new value to config
def set_value(config, key, value):
    config[key] = value

def show_config(config):
    for key in config:
        print(f"-- {key}: {config[key]}")

def ramdon_topo():
    import random
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(4, 4))
    random.seed(11)

    theta = np.linspace(0, 2 * np.pi, 1000)
    x = np.cos(theta)*750
    y = np.sin(theta)*750
    plt.scatter(x, y, color='black', s=0.5**2)

    pos_node = []
    x = []
    y = []
    for i in range(30):
        flag = True
        while flag:
            tmp_x, tmp_y = random.randint(-750,750), random.randint(-750,750)
            if tmp_x**2 + tmp_y**2 <= 750**2:
                flag = False
        x.append(tmp_x)
        y.append(tmp_y)
        pos_node.append((x[-1], y[-1]))
    print(pos_node)
    plt.scatter(x, y, s=5**2)
    plt.savefig("topo.png")

def new_topo():
    import random
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(4, 4))
    random.seed(1)

    theta = np.linspace(0, 2 * np.pi, 1000)
    R = 100
    x = np.cos(theta)*R
    y = np.sin(theta)*R
    plt.scatter(x, y, color='black', s=0.5**2)

    # pos_node = [(176, 396), (203, 175), (290, 452), (-362, -372), (298, 224), 
    #             (519, 307), (-369, -558), (164, -129), (-460, -525), (469, 61), 
    #             (177, 589), (-428, 526), (-255, 478), (-503, 205), (-82, 152), 
    #             (460, -350), (313, -272), (561, -148), (13, 431), (-381, 83), 
    #             (378, -580), (-499, -130), (-105, -280), (300, -159), (264, -489), 
    #             (70, -530), (-155, 41), (-313, -321), (-611, 122), (18, 701)]
    # pos_node = [(-298, -792), (-500, 200), (394, 598), (-900, -254), (162, -920),
    #         (-124, 340), (66, 450), (-160, 148), (600, -250), (160, -24),
    #         (-320, -600), (650, -454), (296, -442), (-590, 560), (640, 424),
    #         (-96, -130), (-696, -256), (-336, 390), (100, 640), (-152, 712)]
    pos_node = [(-29, -79), (-50, 20), (39, 59), (-90, -25), (16, -92), 
                    (-12, 34), (6, 45), (-16, 14), (60, -25), (16, -2), 
                    (-32, -60), (65, -45), (29, -44), (-59, 56), (64, 42), 
                    (-9, -13), (-69, -25), (-33, 39), (10, 64), (-15, 71)]
    # pos_node = [(176, 396), (203, 175), (290, 452), (-362, -372), (298, 224), 
    #             (519, 307), (-369, -558), (164, -129), (-460, -525), (469, 61), 
    #             (177, 589), (-428, 526), (-255, 478), (-503, 205), (-82, 152), 
    #             (460, -350), (313, -272), (561, -148), (13, 431), (-381, 83)]
    # pos_node = [(-29, -79), (-50, 20), (39, 59), (-90, -25), (16, -92), 
    #                 (-12, 34), (6, 45), (-16, 14), (60, -25), (16, -2), 
    #                 (-32, -60), (65, -45), (29, -44), (-59, 56), (64, 42), 
    #                 (-9, -13), (-69, -25), (-33, 39), (10, 64), (-15, 71)]
    x = []
    y = []
    x_r = []
    y_r = []
    x_g = []
    y_g = []
    for i in range(len(pos_node)):
        if i in [7,9,15]:
            x_r.append(pos_node[i][0])
            y_r.append(pos_node[i][1])
        if i in [5,6,17,19]:
            x_g.append(pos_node[i][0])
            y_g.append(pos_node[i][1])
        x.append(-pos_node[i][0])
        y.append(-pos_node[i][1])
        plt.text(-pos_node[i][0], -pos_node[i][1], str(i))
    plt.scatter(x, y, s=5**2)
    # plt.scatter(x_r, y_r, s=5**2, color='r')
    # plt.scatter(x_g, y_g, s=5**2, color='g')
    plt.savefig("topo.png", bbox_inches='tight', pad_inches=0.05, dpi=600)
    plt.savefig("topo.eps", format='eps', bbox_inches='tight', pad_inches=0.05, dpi=600)


if __name__ == "__main__":
    # init_config_file()
    # show_config(get_config())
    new_topo()
    # ob = [(-298, -792), (-500, 200), (394, 598), (-900, -254), (162, -920),
    #         (-124, 340), (66, 450), (-160, 148), (600, -250), (160, -24),
    #         (-320, -600), (650, -454), (296, -442), (-590, 560), (640, 424),
    #         (-96, -130), (-696, -256), (-336, 390), (100, 640), (-152, 712)]
    # out = []
    # for o in ob:
    #     out.append((int(o[0]/3), int(o[1]/3)))
    # print(out)