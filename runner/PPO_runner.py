from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from env.env import env as RoutingEnv
from model.PPO import PPOCHSeletion as PPO
from utils.log_utils import record_ac_loss, record_object, record_ppo_ratio


class PPORunner(object):
    def __init__(self, config, load_episode=-1) -> None:
        self.env_n = config['env_n']
        self.env_step = config['env_step']
        self.env = [RoutingEnv(config) for _ in range(self.env_n)]
        self.model = PPO(config)
        self.config = config

        # for log
        self.max_round = -1
        self.save_freq = config['save_freq']

        if load_episode != -1:
            self.model.load(load_episode)
            print("load model, episode: " + str(load_episode))

    def model_test(self):
        env = RoutingEnv(self.config)
        done = None
        state = env.get_obs()
        while (not done) and (env.cnt_transmit < self.config['max_step']):
            action, action_prob, probs_entropy = self.model.choose_abstract_action(state)
            reward, done = env.interval_step(action)
            # print(reward)
            state_next = env.get_obs() 
            state = state_next
            if done: break
        return env.cnt_transmit

    def run(self, cnt_episode):
        for _ in range(self.env_step):
            for idx in range(self.env_n):
                state = self.env[idx].get_obs()
                action, action_prob, probs_entropy = self.model.choose_abstract_action(state)
                reward, done = self.env[idx].interval_step(action)
                state_next = self.env[idx].get_obs()
                self.model.store_transition(state, action, action_prob, reward, done, state_next, idx)

                if done:
                    self.env[idx].reset()

        aloss, closs = self.model.update()
        for idx in range(len(aloss)):
            record_ac_loss(aloss[idx], closs[idx], self.config)
        cnt_transmit = self.model_test()

        if cnt_episode % self.save_freq == 0:
            self.model.save(cnt_episode)
        if self.max_round < cnt_transmit:
            self.model.save_best(cnt_episode) 
            self.max_round = cnt_transmit
        record_object(cnt_transmit, self.config)
        print(f"update step: {cnt_episode}, lifetime: {cnt_transmit}")       

