from ebrp.EBRP import ebrp
from env.node import NodeRL as Node
import numpy as np


class env:
    def __init__(self, config):
        self.config = config
        pos_node = np.array(self.config['pos_node'])
        self.num_node = len(pos_node)
        self.node = [Node(i, pos_node[i], self.config['sensor_energy'], \
            self.config['unit_cost_rec'], self.config['unit_cost_send'], \
            self.config['unit_cost_proc'], self.config['died_threshold']
            ) for i in range(self.num_node)]
        self.idx_sink = self.num_node - 1
        self.proc_inter = ebrp(config) # inter-cluster protocol
        assert self.set_neighbor()

        self.state_dim = self.config['state_dim']
        self.action_dim = self.num_node
        self.round_baseline = self.config['round_baseline']
        self.trans_in_interval = self.config['trans_in_interval']
        self.round_base = self.config['round_base']
        self.data_max = self.config['data_max']
        self.data_min = self.config['data_min']
        self.map_range = self.config['map_range']

        self.cnt_transmit = 0
        self.pos_hard_code = self.get_pos_hard_code()
        # self.recv_consomed_en = [0 for _ in range(self.num_node)]
        # self.send_consomed_en = [0 for _ in range(self.num_node)]

    def get_node_energy(self):
        energy_last = []
        for node in self.node:
            energy_last.append(node.energy)
        return energy_last

    def reset(self):
        for i in range(self.num_node):
            self.node[i].reset()
        self.cnt_transmit = 0

    '''广播建立邻居表，设定不消耗能量，仅用于辅助'''
    def set_neighbor(self):
        neighbors = [[] for _ in range(self.num_node)]
        for i in range(self.num_node):
            tmp = 0
            for j in range(self.num_node):
                if i != j:
                    dist = np.sqrt(np.sum(np.power(self.node[i].pos - self.node[j].pos, 2)))
                    if dist <= self.config['comm_radius']:
                        neighbors[i].append({'nodeId': j, 'dist': dist})
                        tmp += 1
            if tmp == 0:
                return False
        for i in range(self.num_node):
            self.node[i].set_neighbor(neighbors[i])
        return True
    
    def interval_step(self, idx_sink, count=-1):
        self.reset_node_data_histo()
        action = self.proc_inter.get_route(self.node, idx_sink)
        if count == 100 or count == 200:
            print(f"action: {action}")
        self.idx_sink = idx_sink
        self.set_route(action)
        last_energy = self.get_node_energy()
        for _ in range(self.trans_in_interval):
            done = self.transmit()
            if done: break
            self.cnt_transmit += 1
        # return self.get_reward_sparse(done), done
        return self.get_reward_min_energy(done, last_energy) + self.get_reward_sparse(done), done
        # return self.get_reward_min_energy(done, last_energy), done
    
    def get_reward_sparse(self, done):
        if not done:
            reward = 0
        else:
            reward = (self.cnt_transmit - self.round_baseline) / self.round_base
            # reward = (self.cnt_transmit) / self.round_base
        return reward

    def get_reward_min_energy(self, done, last_energy):
        max_en = self.config['energy_max']
        min_en = self.config['energy_min']
        if done:
            return 0
        en = self.get_node_energy()
        # max_en_consume = max([last_energy[i] - en[i] for i in range(self.num_node)])
        # normalize_en = -((max_en_consume - min_en)/(max_en - min_en) *2 - 1)
        en_consume = sum([last_energy[i] - en[i] for i in range(self.num_node)])
        normalize_en = -((en_consume - min_en)/(max_en - min_en) *2 - 1)
        normalize_en = min(2, normalize_en)
        normalize_en = max(-2, normalize_en)
        return normalize_en

    def getDistBetweenNodes(self, i, j):
        return np.sqrt(np.sum(np.power(self.node[i].pos - self.node[j].pos, 2)))

    def transmit(self):
        packets = self.create_packet()
        done = False
        for i in range(self.num_node):  # 在时间允许的情况下，逐个传输
            assert self.node[i].next_hop != -1
            this_node = i
            next_node = self.node[i].next_hop
            while this_node != self.idx_sink:
                if this_node == i:
                    self.node[this_node].my_data_size += packets[i]
                else:
                    self.node[this_node].others_date_size += packets[i]

                done = self.node[this_node].send_packet(packets[i]) or done
                # self.send_consomed_en[this_node] += \
                #     packets[i] * self.node[this_node].tx_dist * self.node[this_node].unit_cost_send_dist**2
                done = self.node[next_node].receive_packet(packets[i]) or done
                # self.recv_consomed_en[next_node] += \
                #     packets[i] * self.node[next_node].unit_cost_rec
                this_node = next_node
                if not this_node:
                    break
                next_node = self.node[this_node].next_hop
        size = sum(packets)
        self.node[self.idx_sink].energy -= size * self.node[self.idx_sink].unit_cost_send_dist \
            * self.config['collected_dist'] ** 2
        done = self.node[self.idx_sink].energy < self.node[self.idx_sink].died_threshold or done
        return done

    def reset_node_data_histo(self):
        for i in range(self.num_node):
            self.node[i].my_data_size = 0
            self.node[i].others_date_size = 0

    def set_route(self, table):  # table:[(nodeId, nextHop), ..]
        for item in table:
            x, y = item[0], item[1]
            self.node[x].set_next_hop(y)

    def create_packet(self):
        return np.random.randint(self.data_min, self.data_max, self.num_node)

    def get_obs(self):
        obs = []
        for idx, pos in enumerate(self.pos_hard_code):
            obs.append(self.node[idx].energy/0.15)
            # obs.append(pos[0])
            # obs.append(pos[1])
        return np.array(obs)

    def get_adj_matrix(self):
        ret = [[0]*self.num_node for _ in range(self.num_node)]
        for i in range(self.num_node):
            for nbr in self.node[i].neighbor_list:
                if nbr['nodeId'] != i:
                    ret[i][nbr['nodeId']] = 1
        return np.array(ret)

    def get_pos_hard_code(self):
        pos = [[self.node[i].pos[0] / self.map_range, self.node[i].pos[1] / self.map_range] for i in range(self.num_node)]
        return pos

    def get_dist_between_nodes(self, i, j):
        return np.sqrt(np.sum(np.power(self.node[i].pos - self.node[j].pos, 2)))
