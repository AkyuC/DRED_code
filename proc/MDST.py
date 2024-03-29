#!/usr/bin/env python3
from collections import defaultdict, namedtuple


Arc = namedtuple('Arc', ('tail', 'weight', 'head')) # tail <--- head

def convert_matrix(matrix, neigbor_mask):
    arcs = []
    # 对一个节点的势能差值进行放缩，放缩到[0,1]之间
    for i in range(len(matrix[0])):
        tmp = []
        for j in range(len(matrix[i])):
            if neigbor_mask[i][j] == 1 and i != j:
                tmp.append(matrix[i][j])
        min_potential = min(tmp)
        max_potential = max(tmp)
        for j in range(len(matrix[i])):
            if neigbor_mask[i][j] == 1 and i != j:
                matrix[i][j] = (matrix[i][j] - min_potential)/(max_potential - min_potential)
                # 因为下面是最小有向树，所有需要取负值，变成最大，并且需要反向，因为原算法是树往下的，我们需要往上聚合
                arcs.append(Arc(i, -matrix[i][j], j))
    return arcs

def min_spanning_arborescence(matrix, neigbor_mask, sink):
    arcs = convert_matrix(matrix, neigbor_mask)
    good_arcs = []
    quotient_map = {arc.tail: arc.tail for arc in arcs}
    quotient_map[sink] = sink
    while True:
        min_arc_by_tail_rep = {}
        successor_rep = {}
        for arc in arcs:
            if arc.tail == sink:
                continue
            tail_rep = quotient_map[arc.tail]
            head_rep = quotient_map[arc.head]
            if tail_rep == head_rep:
                continue
            if tail_rep not in min_arc_by_tail_rep or min_arc_by_tail_rep[tail_rep].weight > arc.weight:
                min_arc_by_tail_rep[tail_rep] = arc
                successor_rep[tail_rep] = head_rep
        cycle_reps = find_cycle(successor_rep, sink)
        if cycle_reps is None:
            good_arcs.extend(min_arc_by_tail_rep.values())
            return reverse_direct(spanning_arborescence(good_arcs, sink))
        good_arcs.extend(min_arc_by_tail_rep[cycle_rep] for cycle_rep in cycle_reps)
        cycle_rep_set = set(cycle_reps)
        cycle_rep = cycle_rep_set.pop()
        quotient_map = {node: cycle_rep if node_rep in cycle_rep_set else node_rep for node, node_rep in quotient_map.items()}


def find_cycle(successor, sink):
    visited = {sink}
    for node in successor:
        cycle = []
        while node not in visited:
            visited.add(node)
            cycle.append(node)
            node = successor[node]
        if node in cycle:
            return cycle[cycle.index(node):]
    return None


def spanning_arborescence(arcs, sink):
    arcs_by_head = defaultdict(list)
    for arc in arcs:
        if arc.tail == sink:
            continue
        arcs_by_head[arc.head].append(arc)
    solution_arc_by_tail = {}
    stack = arcs_by_head[sink]
    while stack:
        arc = stack.pop()
        if arc.tail in solution_arc_by_tail:
            continue
        solution_arc_by_tail[arc.tail] = arc
        stack.extend(arcs_by_head[arc.tail])
    return solution_arc_by_tail

def reverse_direct(solution_arc_by_tail):
    ret = []
    for key in solution_arc_by_tail:
        ret.append((key, solution_arc_by_tail[key].head))
    return ret