## DRED
PPO 加了tricks

### 1. 训练版本
v1.0 非稀疏reward，通信半径R=100，reward1, actor_lr=1e-5，critic_lr=1e-4, update time=4, clip_param=0.15, init_energy=0.15, round_base=1800, 

v1.1 非稀疏reward，通信半径R=100，reward1, actor_lr=1e-5，critic_lr=1e-4, update time=4, clip_param=0.15, init_energy=0.15, round_base=1800, 只使用 EBRP值和MDST算法计算路由，不从外往里选择, state_dim=20

v1.2 非稀疏reward，通信半径R=100，reward1, actor_lr=1e-5，critic_lr=1e-4, update time=4, clip_param=0.15, init_energy=0.15, round_base=1800, WSN使用EAR算法

v1.3 非稀疏reward，通信半径R=100，reward1, actor_lr=1e-5，critic_lr=1e-4, update time=4, clip_param=0.15, init_energy=0.15, round_base=1800, 只使用 EBRP值和MDST算法计算路由，不从外往里选择

v1.4 非稀疏reward，通信半径R=100，reward1, actor_lr=1e-5，critic_lr=1e-4, update time=4, clip_param=0.15, init_energy=0.15, round_base=1800, WSN使用EAR算法,加快采样时间

v1.5 非稀疏reward，通信半径R=70，reward1, actor_lr=1e-5，critic_lr=1e-4, update time=4, clip_param=0.15, init_energy=0.15, round_base=1800, 只使用 EBRP值和MDST算法计算路由，不从外往里选择, state_dim=60

v1.6 非稀疏reward，通信半径R=100，reward1, actor_lr=1e-5，critic_lr=1e-4, update time=4, clip_param=0.15, init_energy=0.15, round_base=1800, 只使用 EBRP值和MDST算法计算路由，不从外往里选择, state_dim=60

v1.7 非稀疏reward，通信半径R=130，reward1, actor_lr=1e-5，critic_lr=1e-4, update time=4, clip_param=0.15, init_energy=0.15, round_base=1800, 只使用 EBRP值和MDST算法计算路由，不从外往里选择, state_dim=60

v1.8 非稀疏reward，通信半径R=100，reward1, actor_lr=1e-5，critic_lr=1e-4, update time=4, clip_param=0.15, init_energy=0.15, round_base=1800, 只使用 EBRP值和MDST算法计算路由，不从外往里选择，决策间隔为 5, state_dim=20

v1.9 非稀疏reward，通信半径R=100，reward1, actor_lr=1e-5，critic_lr=1e-4, update time=4, clip_param=0.15, init_energy=0.15, round_base=1800, 只使用 EBRP值和MDST算法计算路由，不从外往里选择，决策间隔为 15, state_dim=20

v1.10 非稀疏reward，通信半径R=70，reward1, actor_lr=1e-5，critic_lr=1e-4, update time=4, clip_param=0.15, init_energy=0.15, round_base=1800, 只使用 EBRP值和MDST算法计算路由，不从外往里选择, state_dim=20

v1.11 非稀疏reward，通信半径R=130，reward1, actor_lr=1e-5，critic_lr=1e-4, update time=4, clip_param=0.15, init_energy=0.15, round_base=1800, 只使用 EBRP值和MDST算法计算路由，不从外往里选择, state_dim=20