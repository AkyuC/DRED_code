import numpy as np


class Node:
    def __init__(self, node_id, pos, init_enery, \
                unit_cost_rec, unit_cost_send_dist, unit_cost_proc, died_threshold):
        self.node_id = node_id  # id标识，19号是sink
        self.pos = np.array(pos)
        self.depth = -1
        self.neighbor_list = []  # [邻居id、距离、剩余能量(可不可以获取)] 距离好像也用不太到
        self.init_energy = init_enery
        self.unit_cost_rec = unit_cost_rec  # 接收数据的单位能量消耗/bit
        self.unit_cost_send_dist = unit_cost_send_dist
        self.unit_cost_proc = unit_cost_proc  # 处理数据的单位能量消耗/bit
        self.died_threshold = died_threshold  # 节点的死亡阈值，当能量低于该阈值时，节点被视为死亡，当网络中有一个节点死亡时，整个WSN瘫痪
        self.reset()

    def reset(self):
        self.energy = self.init_energy
        self.next_hop = None
        self.tx_dist = None

    def set_neighbor(self, neighbors):
        self.neighbor_list = neighbors

    def receive_packet(self, size):
        self.energy -= size * self.unit_cost_rec
        return self.energy < self.died_threshold

    def send_packet(self, size):
        assert self.next_hop != -1
        self.energy -= size * self.unit_cost_send_dist * self.tx_dist ** 2
        return self.energy < self.died_threshold

    def process_packet(self, size):
        self.energy -= size * self.unit_cost_proc
        return self.energy < self.died_threshold

    def set_next_hop(self, next_hop):
        flag = 0
        for nbr in self.neighbor_list:
            if nbr['nodeId'] == next_hop:
                flag = 1
                self.tx_dist = nbr['dist']
                break
        assert flag == 1
        self.next_hop = next_hop
        
        
class NodeRL(Node):
    def __init__(self, node_id, pos, init_enery, \
                unit_cost_rec, unit_cost_send_dist, unit_cost_proc, died_threshold):
        super(NodeRL, self).__init__(node_id, pos, init_enery, \
            unit_cost_rec, unit_cost_send_dist, unit_cost_proc, died_threshold)
        self.my_data_size = 0  # 本决策时隙自己产生的数据
        self.others_date_size = 0  # 本决策时隙帮其他节点转发的数据
        self.prev_action = 0  # 上一次决策的动作

    def reset(self):
        super(NodeRL, self).reset()
        self.reset_info_rl()

    def reset_info_rl(self):
        self.my_data_size = 0  # 本决策时隙自己产生的数据
        self.others_date_size = 0  # 本决策时隙帮其他节点转发的数据
        self.prev_action = 0  # 上一次决策的动作


class Packet:
    def __init__(self):
        pass
