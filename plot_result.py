import os
from utils.log_utils import mkdir
from utils.alg_utils import init
from utils.plot_utils import get_avg_lifetime_max, plot_loss, plot_mean_survival_time, plot_ppo_ratio, plot_ppo_state_value, plot_survival_time

def plot_static():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import font_manager
    my_font = font_manager.FontProperties(fname="./TimesNewRoman.ttc")

    # data_mean = [1256.2, 1322.5, 1238.3, 1603.5, 1440.7, 
    #         1326.7, 1292.6, 1308.4, 1460.3, 1308.9, 
    #         1554.4, 1525.2, 1713.4, 1170.3, 1235.4, 
    #         1344.3, 1609.9, 1306.4, 1261.3, 1294.1,
    #         1187.4, 2619.7, 2513.9, ]
    data_mean = [395.8, 1514.6,
             2179.7, 2436.3, ]
    # data_max = [1276, 1445, 1317, 1626, 1455, 
    #         1416, 1372, 1379, 1518, 1422, 
    #         1585, 1555, 1794, 1279, 1335, 
    #         1429, 1680, 1394, 1337, 1404,
    #         1325, 2641, 2614, ]
    data_max = [398, 1637,
            2228, 2478, ]
    data_max_confidence = np.array(data_max)- np.array(data_mean)
    # data_min = [1237, 1238, 1038, 1579, 1418, 
    #         1257, 1127, 1255, 1386, 1129, 
    #         1526, 1497, 1658, 987,  1116, 
    #         1247, 1488, 1209, 1185, 1089,
    #         1059, 2582, 2428,]
    data_min = [391, 1332,
            2158, 2398,]
    data_min_confidence = np.array(data_mean)- np.array(data_min)
    # labels = ['n1', 'n2', 'n3', 'n4', 'n5', 
    #         'n6', 'n7', 'n8', 'n9', 'n10', 
    #         'n11', 'n12', 'n13', 'n14', 'n15', 
    #         'n16', 'n17', 'n18', 'n19', 'n20',  
    #         'Random', 'Greedy', 'DRED', ]
    labels = ['Static',
            'Random', 'Greedy', 'DRED', ]

    plt.rcParams['figure.figsize'] = (7,5)
    plt.rcParams['font.size'] = 16

    x = np.linspace(1,4,4)
    plt.errorbar(x, data_mean, yerr=[data_min_confidence, data_max_confidence], 
            fmt='o:',ecolor='red',elinewidth=5,ms=7.5,mfc='black',mec='red',capsize=10,color='black')
    plt.text(x[0]+0.05, data_mean[0], '%d' % data_mean[0], color='black', fontdict={'fontsize':14})
    plt.text(x[1]+0.05, data_mean[1]-100, '%d' % data_mean[1], color='black', fontdict={'fontsize':14})
    plt.text(x[2]+0.05, data_mean[2]-150, '%d' % data_mean[2], color='black', fontdict={'fontsize':14})
    plt.text(x[3]-0.15, data_mean[3]-200, '%d' % data_mean[3], color='black', fontdict={'fontsize':14})
#     plt.text(x[4]+0.15, data_mean[4]-150, '%d' % data_mean[4], color='black', fontdict={'fontsize':14})
#     plt.text(x[5]+0.15, data_mean[5], '%d' % data_mean[5], color='black', fontdict={'fontsize':14})
#     plt.text(x[6]-0.05, data_mean[6]-175, '%d' % data_mean[6], color='black', fontdict={'fontsize':14})
#     plt.text(x[7]-0.2, data_mean[7]-200, '%d' % data_mean[7], color='black', fontdict={'fontsize':14})
    plt.ylabel("Number of Lifetime / Rounds", fontdict={'size':14})
    plt.xlabel("Scheme", fontdict={'size':14})
    plt.xticks(x, labels, size = 13.5)
    plt.tick_params(labelsize=13.5)
    plt.savefig('compare_with_static.png', bbox_inches='tight', pad_inches=0.05, dpi=600)
    plt.savefig('compare_with_static.eps', format='eps', bbox_inches='tight', pad_inches=0.05, dpi=600)
    plt.close('all')

def plot_total_energy():
    import matplotlib.pyplot as plt
    import numpy as np
    en_total_greedy = np.loadtxt("Greedy_total_energy")
    en_total_random = np.loadtxt("Random_total_energy")
    en_total_DRED = np.loadtxt("Ours_total_energy")
    en_total_n13 = np.loadtxt("Static_n4_total_energy")
    plt.rcParams['figure.figsize'] = (7,5)
    plt.rcParams['font.size'] = 16
    plt.plot(np.arange(len(en_total_DRED)), en_total_DRED, color="red", label="DRED", alpha = 0.75, linewidth = 2)
    plt.plot(np.arange(len(en_total_random)), en_total_random, color='black', label="Random", alpha = 0.75, linewidth = 2)
    plt.plot(np.arange(len(en_total_greedy)), en_total_greedy, color='green', label="Greedy", alpha = 0.75, linewidth = 2)
    plt.plot(np.arange(len(en_total_n13)), en_total_n13, color='b', label="Static", alpha = 0.75, linewidth = 2)
    plt.xticks(np.arange(len(en_total_DRED)), np.arange(len(en_total_DRED))*10)
    plt.ylabel("Total Network Energy / J")
    plt.xlabel("Network Lifetime / Rounds")
    plt.xticks([0,50,100,150,200,250], [0,500,1000,1500,2000,2500])
    # plt.title('lifetime')
    plt.legend()
    plt.savefig('compare_total_energy.png', bbox_inches='tight', pad_inches=0.05, dpi=600)
    plt.savefig('compare_total_energy.eps', format='eps', bbox_inches='tight', pad_inches=0.05, dpi=600)
    plt.close('all')
    

def plot_from_file():
    config = init()
    config['comm_radius'] = 100
    config['ebrp_estimate_radius'] = 100
    config['ver'] = '5.1'
    config['actor_lr'] = 1e-5
    config['critic_lr'] = 1e-4
    config['batch_size'] = 32
    config['ebrp_alpha'] = 0.1
    config['ebrp_beta'] = 0.8
    
    seedset = [1,2,3,4,5]
#     for seed in seedset:
#         config['seed'] = seed
#         mkdir(config, False)
#         plot_loss(config)
#         plot_survival_time(config)
#         plot_ppo_ratio(config)
#         plot_ppo_state_value(config)
    plot_mean_survival_time(config, seedset, 350000)

def get_avg_life_time_max():
    config = init()
    config['comm_radius'] = 100
    config['ebrp_estimate_radius'] = 100
    config['ver'] = '5.1'
    config['actor_lr'] = 1e-5
    config['critic_lr'] = 1e-4
    config['batch_size'] = 32
    config['ebrp_alpha'] = 0.1
    config['ebrp_beta'] = 0.8
    seedset = [1,2,3,4,5]
    for seed in seedset:
        config['seed'] = seed
        print(f"seed{seed}: {get_avg_lifetime_max(config)}")
        

if __name__ == '__main__':
#     plot_from_file()
    plot_static()
    # plot_total_energy()
#     get_avg_life_time_max()