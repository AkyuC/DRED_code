import copy
import numpy as np

from ebrp.MDST import min_spanning_arborescence

class ebrp(object):
    def __init__(self, config):
        self.config = config
        self.alpha = config['ebrp_alpha']
        self.beta = config['ebrp_beta']
        self.es_radius = config['ebrp_estimate_radius']

    def get_route(self, node, idxSink):
        self.node = node
        self.idxSink = idxSink
        self.numNode = len(node)

        self.resetUF()
        self.updateMatrix()
        try:
            return self.calcRoute()
        except AssertionError:
            neigbor_mask = np.zeros((self.numNode, self.numNode))
            for i in range(self.numNode):
                for nbr in self.node[i].neighbor_list:
                    neigbor_mask[i][nbr["nodeId"]] = 1
            return min_spanning_arborescence(self.combineDiff, neigbor_mask, idxSink)   

    # 广度优先搜索计算节点深度
    def calcDepth(self):
        ret = np.zeros(self.numNode, dtype=np.int8)
        q = [self.idxSink]
        layerCnt = 1
        while q:
            tmp = copy.deepcopy(q)
            q = []
            for i in tmp:
                ret[i] = layerCnt
                for nbr in self.node[i].neighbor_list:
                    if (nbr["nodeId"] not in tmp) and (nbr["nodeId"] not in q) and (ret[nbr["nodeId"]] == 0):
                        q.append(nbr["nodeId"])
            layerCnt += 1
        return ret - 1

    def calcDepthDiff(self):
        ret = np.ones((self.numNode, self.numNode), dtype=np.float64)
        ret *= -1
        for i in range(self.numNode):
            for nbr in self.node[i].neighbor_list:
                j = nbr["nodeId"]
                if self.depthList[i] - self.depthList[j] == -1:
                    ret[i, j] = -1 / (self.depthList[i] + 2)
                elif self.depthList[i] - self.depthList[j] == 0:
                    ret[i, j] = 0
                elif self.depthList[i] - self.depthList[j] == 1:
                    ret[i, j] = 1 / (self.depthList[i] + 1)
        return ret

    def calcEnergyDiff(self):  # sink能量量级处理
        t = np.zeros((self.numNode, self.numNode), dtype=np.float64)
        for i in range(self.numNode):
            for nbr in self.node[i].neighbor_list:
                j = nbr["nodeId"]
                t[i, j] = self.node[j].energy / self.node[i].energy
        return self.diffNormalization(t)

    def calcEnergyDensityDiff(self):
        ed = [self.calcEnergyDensityNode(i) for i in range(self.numNode)]
        t = np.zeros((self.numNode, self.numNode), dtype=np.float64)
        for i in range(self.numNode):
            for nbr in self.node[i].neighbor_list:
                j = nbr["nodeId"]
                t[i, j] = ed[j] / ed[i]
        return self.diffNormalization(t)

    def calcCombineDiff(self):
        ret = (1 - self.alpha - self.beta) * self.depthDiffList + \
              self.alpha * self.energyDensityDiffList + self.beta * self.energyDiffList
        for i in range(len(ret)):
            for j in range(len(ret[0])):
                if i != j:
                    ret[i, j] /= self.getDistBetweenNodes(i, j)
        return ret

    def diffNormalization(self, diffMatrix):
        ret = np.ones((self.numNode, self.numNode), dtype=np.float64)
        ret *= -1
        for i in range(len(diffMatrix)):
            for j in range(len(diffMatrix[0])):
                if diffMatrix[i, j] < 1:
                    ret[i, j] = diffMatrix[i, j] - 1
                else:
                    ret[i, j] = 1 - 1 / diffMatrix[i, j]
        return ret

    def calcEnergyDensityNode(self, nodeId):
        ret = self.node[nodeId].energy
        for nbr in self.node[nodeId].neighbor_list:
            j = nbr["nodeId"]
            if self.getDistBetweenNodes(nodeId, j) <= self.es_radius:
                ret += self.node[j].energy
        return ret

    def updateMatrix(self):
        self.resetUF()
        self.depthList = self.calcDepth()
        self.depthDiffList = self.calcDepthDiff()
        self.energyDiffList = self.calcEnergyDiff()
        self.energyDensityDiffList = self.calcEnergyDensityDiff()
        self.combineDiff = self.calcCombineDiff()

    def calcRoute(self):
        ret = []
        # 决策顺序：按距离sink的远近
        tmp = []
        for i in range(self.numNode):
            if i != self.idxSink:
                tmp.append((i, self.getDistBetweenNodes(i, self.idxSink)))
        nodeQueue = sorted(tmp, key=lambda x: x[1], reverse=True)
        self.resetUF()
        for nodeId in nodeQueue:
            nodeId = nodeId[0]
            tmp = []
            for nbr in self.node[nodeId].neighbor_list:
                tmp.append((nbr["nodeId"], self.combineDiff[nodeId, nbr["nodeId"]]))
            parentQueue = sorted(tmp, key=lambda x: x[1], reverse=True)
            for parent in parentQueue:
                parent = parent[0]
                if self.checkInsertEdge((parent, nodeId)):
                    ret.append((nodeId, parent))
                    break
        assert len(ret) == self.numNode - 1
        return ret

    # 判断edge端点是否未连通，返回True则添加该edge
    def checkInsertEdge(self, edge):
        if self.UF_leader[edge[0]] == -1 and self.UF_leader[edge[1]] == -1:
            self.UF_leader[edge[0]], self.UF_leader[edge[1]] = edge[0], edge[0]
        elif self.UF_leader[edge[0]] == -1:
            self.UF_leader[edge[0]] = self.UF_leader[edge[1]]
        elif self.UF_leader[edge[1]] == -1:
            self.UF_leader[edge[1]] = self.UF_leader[edge[0]]
        else:
            if self.UF_leader[edge[0]] != self.UF_leader[edge[1]]:
                tmp = self.UF_leader[edge[1]]
                for i in range(len(self.UF_leader)):
                    if self.UF_leader[i] == tmp:
                        self.UF_leader[i] = self.UF_leader[edge[0]]
            else:
                return False
        return True

    def resetUF(self):
        self.UF_leader = [-1 for _ in range(self.numNode)]

    def getDistBetweenNodes(self, i, j):
        return np.sqrt(np.sum(np.power(self.node[i].pos - self.node[j].pos, 2)))
