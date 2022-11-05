## DRED
PPO 加了tricks

### 1. 训练版本
熵正则加错了，v5.0 非稀疏reward，通信半径R=100， reward1, actor_lr=1e-5，critic_lr=1e-4, update time=4, clip_param=0.15, init_energy=0.15, round_base=2000, round_base=400<br>
v5.1 非稀疏reward，通信半径R=100， reward3, actor_lr=1e-5，critic_lr=1e-4, update time=4, clip_param=0.15, init_energy=0.15, round_base=1800, round_base=400, maxth=0.05, minth=0.015