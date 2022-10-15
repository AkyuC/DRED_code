from pathlib import Path
import os


def mkdir(config, record=True):
    alg = config['alg']
    seed = config['seed']
    if record == True:
        path = f'./{alg}_outcome'
    else:
        path = f'./{alg}_plot'
    if not (Path(path)).is_dir():
        os.mkdir(path)
    path += "/seed" + str(seed)
    if not (Path(path)).is_dir():
        os.mkdir(path)
    path += f"/v{config['ver']}_a{config['ebrp_alpha']}_b{config['ebrp_beta']}_alr{config['actor_lr']}_clr{config['critic_lr']}_bs{config['batch_size']}_cr{config['comm_radius']}_esr{config['ebrp_estimate_radius']}"
    if not (Path(path)).is_dir():
        os.mkdir(path)
    if not (Path(path + "/loss")).is_dir():
        os.mkdir(path + "/loss")
    if not (Path(path + "/model")).is_dir():
        os.mkdir(path + "/model")
    if not (Path(path + "/reward")).is_dir():
        os.mkdir(path + "/reward")
    if not (Path(path + "/object")).is_dir():
        os.mkdir(path + "/object")

def get_loss_path(config, record=True):
    alg = config['alg']
    seed = config['seed']
    if record == True:
        path = f'./{alg}_outcome/seed{seed}'
    else:
        path = f'./{alg}_plot/seed{seed}'
    path += f"/v{config['ver']}_a{config['ebrp_alpha']}_b{config['ebrp_beta']}_alr{config['actor_lr']}_clr{config['critic_lr']}_bs{config['batch_size']}_cr{config['comm_radius']}_esr{config['ebrp_estimate_radius']}"
    path += f"/loss"
    return path

def get_reward_path(config, record=True):
    alg = config['alg']
    seed = config['seed']
    if record == True:
        path = f'./{alg}_outcome/seed{seed}'
    else:
        path = f'./{alg}_plot/seed{seed}'
    path += f"/v{config['ver']}_a{config['ebrp_alpha']}_b{config['ebrp_beta']}_alr{config['actor_lr']}_clr{config['critic_lr']}_bs{config['batch_size']}_cr{config['comm_radius']}_esr{config['ebrp_estimate_radius']}"
    path += f"/reward"
    return path

def get_model_path(config, record=True):
    alg = config['alg']
    seed = config['seed']
    if record == True:
        path = f'./{alg}_outcome/seed{seed}'
    else:
        path = f'./{alg}_plot/seed{seed}'
    path += f"/v{config['ver']}_a{config['ebrp_alpha']}_b{config['ebrp_beta']}_alr{config['actor_lr']}_clr{config['critic_lr']}_bs{config['batch_size']}_cr{config['comm_radius']}_esr{config['ebrp_estimate_radius']}"
    path += f"/model"
    return path

def get_object_path(config, record=True):
    alg = config['alg']
    seed = config['seed']
    if record == True:
        path = f'./{alg}_outcome/seed{seed}'
    else:
        path = f'./{alg}_plot/seed{seed}'
    path += f"/v{config['ver']}_a{config['ebrp_alpha']}_b{config['ebrp_beta']}_alr{config['actor_lr']}_clr{config['critic_lr']}_bs{config['batch_size']}_cr{config['comm_radius']}_esr{config['ebrp_estimate_radius']}"
    path += f"/object"
    return path

def record_ac_loss(aloss, closs, config):
    with open(get_loss_path(config, True) + f'/aloss', 'a+') as file:
        file.write(str(aloss) + '\n')
    with open(get_loss_path(config, True) + '/closs', 'a+') as file:
        file.write(str(closs) + '\n')

def record_dqn_loss(loss, config):
    with open(get_loss_path(config, True) + '/loss', 'a+') as file:
        file.write(str(loss) + '\n')

def record_ppo_ratio(ratio, config):
    with open(get_loss_path(config, True) + '/ratio', 'a+') as file:
        file.write(str(ratio) + '\n')

def record_ppo_state_value(state_value, config):
    with open(get_loss_path(config, True) + '/state_value', 'a+') as file:
        file.write(str(state_value) + '\n')

def record_reward(reward, episode, config):
    with open(get_reward_path(config, True) + f'/episode{episode}_reward', 'a+') as file:
        file.write(str(reward) + '\n')

def record_object(lifetime, config):
    with open(get_object_path(config, True)+'/lifetime', 'a+') as file:
        file.write(str(lifetime) + '\n')