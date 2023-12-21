import copy
import math
import numpy as np

# 能量消耗模型
E_recv = 50e-9
E_send = 1e-12

PARAMS = {"alpha": -0.5,
          "beta": -1,
          "c_thresh": 1}  
# 1 -20 1.5
# 1 -10 5


class ear():
    def __init__(self, config, node):
        self.config = config
        self.node = node
        self.numNode = config["action_dim"]
        # 列表元素为每个节点的邻居集合
        self.nbrSetList = self.getNbrSetList()

    def getNbrSetList(self):
        ret = []
        for i in range(self.numNode):
            nbrSet = set()
            for nbr in self.node[i].neighbor_list:
                nbrSet.add(nbr["nodeId"])
            ret.append(nbrSet)
        return ret

    def getTransTable(self):  # source:i; dest:sink; j->k
        mask = np.zeros((self.numNode, self.numNode, self.numNode), dtype=np.int8)
        for i in range(self.numNode):
            if i == self.idxSink:
                continue
            for j in range(self.numNode):
                for k in self.nbrSetList[j]:
                    if (self.getDistBetweenNodes(j, self.idxSink) > self.getDistBetweenNodes(k, self.idxSink)) \
                            and (self.getDistBetweenNodes(j, i) < self.getDistBetweenNodes(k, i)):
                        mask[i, j, k] = 1
        return mask

    def updateMatrix(self):
        for source in range(self.numNode):
            if source == self.idxSink:
                continue
            q = [self.idxSink]
            while q:
                tmp = copy.deepcopy(q)
                q = []
                for k in tmp:
                    self.updateCost(k)
                    for j in self.nbrSetList[k]:
                        if self.transTable[source, j, k]:
                            q.append(j)
                            m1 = (E_send * self.getDistBetweenNodes(j, k) ** 2 + E_recv) ** PARAMS["alpha"]
                            m2 = (self.node[k].energy / self.node[k].init_energy) ** PARAMS["beta"]
                            self.cMatrix[j, k] = self.costList[k] + m1 * m2
                q = set(q)

    def updateProb(self, nodeId):
        cMin = math.inf
        for i in range(self.numNode):
            if self.cMatrix[nodeId, i] and self.cMatrix[nodeId, i] < cMin:
                cMin = self.cMatrix[nodeId, i]
        if cMin == math.inf:
            return
        tmp = np.zeros(self.numNode, dtype=np.float64)
        for i in range(self.numNode):
            if self.cMatrix[nodeId, i] and self.cMatrix[nodeId, i] <= PARAMS["c_thresh"] * cMin:
                tmp[i] = 1 / self.cMatrix[nodeId, i]
        self.probTable[nodeId, :] = tmp / np.sum(tmp)

    def updateCost(self, nodeId):
        self.updateProb(nodeId)
        # 利用子节点的节点选择概率和对应路径开销加权求和，得到父节点的节点开销
        self.costList[nodeId] = np.sum(np.multiply(self.probTable[nodeId, :], self.cMatrix[nodeId, :]))

    def calcRoute(self):
        ret = []
        for i in range(self.numNode):
            if i == self.idxSink:
                continue
            self.updateProb(i)
            try:
                nextHop = np.random.choice(range(self.numNode), p=self.probTable[i, :])
            except Exception as err:
                print(err)
            ret.append((i, nextHop))
        return ret
    
    def getDistBetweenNodes(self, i, j):
        return np.sqrt(np.sum(np.power(self.node[i].pos - self.node[j].pos, 2)))
    
    def get_route(self, node, idxSink):
        self.idxSink = idxSink
        self.node = node
        self.numNode = len(node)

        # 路由表
        self.transTable = self.getTransTable()  # binary
        # 节点开销
        self.costList = np.zeros(self.numNode, dtype=np.float64)
        # 路径开销
        self.cMatrix = np.zeros((self.numNode, self.numNode), dtype=np.float64)
        # 节点选择概率
        self.probTable = np.zeros((self.numNode, self.numNode), dtype=np.float64)
        
        self.updateMatrix()
        
        return self.calcRoute()