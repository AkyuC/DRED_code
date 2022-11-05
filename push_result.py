import os
from pathlib import Path
from utils.log_utils import get_loss_path, get_object_path
from utils.alg_utils import init


if __name__ == '__main__':
    config = init()
    config['comm_radius'] = 100
    config['ebrp_estimate_radius'] = 100
    config['ver'] = '5.1'
    config['actor_lr'] = 1e-5
    config['critic_lr'] = 1e-4
    config['batch_size'] = 32
    config['ebrp_alpha'] = 0.1
    config['ebrp_beta'] = 0.8
    big_ver = '/v4'
    small_ver = '/v4.8'
    
    seedset = [1,2,3,4,5]

    path = '../iot_reward_shaping' + big_ver
    if not (Path(path)).is_dir():
        os.mkdir(path)
    path += small_ver
    if (Path(path)).is_dir():
        os.system(f'rm -rf {path}')
    os.mkdir(path)

    for seed in seedset:
        config['seed'] = seed
        object_path = get_object_path(config, False)
        loss_path = get_loss_path(config, False)
        seed_path = path + f'/seed{seed}'
        os.mkdir(seed_path)
        os.system(f"cp {object_path}/* {seed_path}")
        os.system(f"cp {loss_path}/* {seed_path}")

    with open(f'{path}/readme.md', 'w+') as f:
        f.truncate()
        # f.write(f'### 0 . seed1\n')
        # f.write('|name|figure|\n')
        # f.write('|:--:|:--:|\n')
        # f.write('|lifetime_avg|![image](https://github.com/ARM-CoMAL/iot_reward_shaping/blob/main' \
        #     + big_ver + small_ver + f'/seed1/lifetime_avg.png)|\n')
        idx = 1
        for seed in seedset:
            f.write(f'### {idx}. seed{seed}\n')
            f.write('|name|figure|\n')
            f.write('|:--:|:--:|\n')
            f.write('|lifetime|![image](https://github.com/ARM-CoMAL/iot_reward_shaping/blob/main' \
                + big_ver + small_ver + f'/seed{seed}/lifetime.png)|\n')
            f.write('|aloss|![image](https://github.com/ARM-CoMAL/iot_reward_shaping/blob/main' \
                + big_ver + small_ver + f'/seed{seed}/aloss.png)|\n')
            f.write('|closs|![image](https://github.com/ARM-CoMAL/iot_reward_shaping/blob/main' \
                + big_ver + small_ver + f'/seed{seed}/closs.png)|\n')
            f.write('|ratio|![image](https://github.com/ARM-CoMAL/iot_reward_shaping/blob/main' \
                + big_ver + small_ver + f'/seed{seed}/ratio.png)|\n')
            f.write('|state value|![image](https://github.com/ARM-CoMAL/iot_reward_shaping/blob/main' \
                + big_ver + small_ver + f'/seed{seed}/state_value.png)|\n')
            f.write('\n')
            idx += 1