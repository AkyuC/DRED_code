from utils.alg_utils import init
from utils.log_utils import mkdir
from runner.PPO_runner import PPORunner 


if __name__ == '__main__':
    config = init()
    mkdir(config, True)
    runner = PPORunner(config, config['load_episode'])

    # rm -rf ./*/*/*v1.5*

    for idx_episode in range(config['max_episode']):
        if config['load_episode'] != -1:
            runner.run(idx_episode + config['load_episode'])
        else:
            runner.run(idx_episode)