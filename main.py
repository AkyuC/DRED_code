from utils.alg_utils import init
from utils.log_utils import mkdir
from runner.PPO_runner import PPORunner 


if __name__ == '__main__':
    config = init()
    mkdir(config, True)
    runner = PPORunner(config)

    for idx_episode in range(config['max_episode']):
        runner.run(idx_episode)