import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
random.seed(1)


# def gen_graph(nodeSize, byzantine):
#     """
#     Randomly generate a graph where the regular workers are connected.

#     :param nodeSize: the number of workers
#     :param byzantine: the set of Byzantine workers
#     """
#     while True:
#         G = nx.fast_gnp_random_graph(nodeSize, 0.5, seed=1)
#         H = G.copy()
#         for i in byzantine:
#             H.remove_node(i)
#         num_connected = 0
#         for _ in nx.connected_components(H):
#             num_connected += 1
#         if num_connected == 1:
#             # nx.draw(G)
#             # plt.show()
#             break
#     return G
#     # G = nx.complete_graph(nodeSize)
#     # return G


def gen_twocastle_graph(k, byzantine):
    """
    There are 2k nodes in the network totally
    """
    assert k >= 3, 'k must be greater than or equal 3'
    assert len(byzantine) <= k - 2, 'Byzantine_size must be less than or equal to k - 2'

    nodesize = 2 * k
    graph = nx.Graph()
    graph.add_nodes_from(range(nodesize))

    # inner edges
    for castle in range(2):
        edge_list = [(i, j) for i in range(k*castle, k*castle+k) for j in range(i+1, k*castle+k)]
        graph.add_edges_from(edge_list)

    # outer edges
    edge_list = [(i, j) for i in range(k) for j in range(k, 2*k) if i + k != j]
    graph.add_edges_from(edge_list)

    # nx.draw(graph)
    # plt.show()

    return graph


# def gen_line_graph(nodesize):



def metropolis_weight(G):
    num = len(G)
    matrix = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            if (i, j) in G.edges() and i != j:
                matrix[i][j] = 1 / (1 + max(G.degree[i], G.degree[j]))

    for i in range(num):
        for j in range(num):
            if i == j:
                matrix[i][j] = 1 - sum(matrix[i])

    return matrix


optConfig = {
    'nodeSize': 12,
    'byzantineSize': 1,

    'iterations': 30000,
    'decayWeight': 0.01,

    'batchSize':32,
}


mnistConfig = {
    'trainNum': 60000,
    'testNum': 10000,
    'dimensions': 784,
    'classes': 10,
}

fashionmnistConfig = mnistConfig.copy()

DPSGDConfig = optConfig.copy()
DPSGDConfig['learningStep'] = 0.1       # without attack
# DPSGDConfig['learningStep'] = 0.18      # same-value attack
# DPSGDConfig['learningStep'] = 0.5       # sign-flipping attack
# DPSGDConfig['learningStep'] = 0.4         # Non-iid setting

ByRDiEConfig = optConfig.copy()
ByRDiEConfig['learningStep'] = 1      # without attack
# ByRDiEConfig['learningStep'] = 0.18      # same-value attack
# ByRDiEConfig['learningStep'] = 0.8       # sign-flipping attack
# ByRDiEConfig['learningStep'] = 0.9       # Non-iid setting

BRIDGEConfig = optConfig.copy()
BRIDGEConfig['learningStep'] = 0.1     # without attack
# BRIDGEConfig['learningStep'] = 0.9       # same-value attack
# BRIDGEConfig['learningStep'] = 0.6       # sign-flipping attack
# BRIDGEConfig['learningStep'] = 0.1       # Non-iid setting

# DrsaConfig = optConfig.copy()
# DrsaConfig['learningStep'] = 0.3
# DrsaConfig['penaltyPara'] = 0.005         # without attack

# DrsaConfig['learningStep'] = 0.28
# DrsaConfig['penaltyPara'] = 0.01        # same-value attack

# DrsaConfig['learningStep'] = 0.5
# DrsaConfig['penaltyPara'] = 0.0022        # sign-flipping attack

# DrsaConfig['learningStep'] = 0.4
# DrsaConfig['penaltyPara'] = 0.02          # Non-iid setting

DrsaConfig = optConfig.copy()
DrsaConfig['learningStep'] = 0.01
DrsaConfig['penaltyPara'] = 0.5

DrsaSAGAConfig = optConfig.copy()
DrsaSAGAConfig['learningStep'] = 0.005
DrsaSAGAConfig['penaltyPara'] = 0.005

DrsaLSVRGConfig = optConfig.copy()
DrsaLSVRGConfig['learningStep'] = 0.01
DrsaLSVRGConfig['penaltyPara'] = 0.005


DrsaGdConfig = optConfig.copy()
DrsaGdConfig['learningStep'] = 0.1
DrsaGdConfig['penaltyPara'] = 0.02
DrsaGdConfig['iterations'] = 20000

# DrsaSpiderConfig = optConfig.copy()
# DrsaSpiderConfig['learningStep'] = 0.01
# DrsaSpiderConfig['penaltyPara'] = 0.005
#
# DrsaKatyushaConfig = optConfig.copy()
# DrsaKatyushaConfig['learningStep'] = 0.01
# DrsaKatyushaConfig['penaltyPara'] = 0.005

GDConfig = optConfig.copy()
GDConfig['learningStep'] = 0.001
GDConfig['iterations'] = 100000

# randomly generate Byzantine workers
byzantine = random.sample(range(optConfig['nodeSize']), optConfig['byzantineSize'])  # 随机选取错误节点
regular = list(set(range(optConfig['nodeSize'])).difference(byzantine))  # 正常节点的集合

# generate topology graph
# G = gen_graph(optConfig['nodeSize'], byzantine)
G = gen_twocastle_graph(optConfig['nodeSize'] // 2, byzantine)

# generate weight matrix according to metropolis weight
weight_matrix = metropolis_weight(G)

