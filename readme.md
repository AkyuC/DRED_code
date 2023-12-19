## DRED
PPO 加了tricks

### 1. 训练版本
v1.0 非稀疏reward，通信半径R=100，reward1, actor_lr=1e-5，critic_lr=1e-4, update time=4, clip_param=0.15, init_energy=0.15, round_base=1800, 

v1.1 非稀疏reward，通信半径R=100，reward1, actor_lr=1e-5，critic_lr=1e-4, update time=4, clip_param=0.15, init_energy=0.15, round_base=1800, 只使用 EBRP值和MDST算法计算路由，不从外往里选择

v1.2 非稀疏reward，通信半径R=100，reward1, actor_lr=1e-5，critic_lr=1e-4, update time=4, clip_param=0.15, init_energy=0.15, round_base=1800, WSN使用EAR算法

v1.1 非稀疏reward，通信半径R=100，reward1, actor_lr=1e-5，critic_lr=1e-4, update time=4, clip_param=0.15, init_energy=0.15, round_base=1800, 只使用 EBRP值和MDST算法计算路由，不从外往里选择