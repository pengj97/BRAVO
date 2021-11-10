# import os
# import sys
# os.chdir(sys.path[0])
# sys.path.append("../../")
import numpy as np
import networkx as nx
import random
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

config = {
    'nodeSize': 4,
    'byzantineSize': 1,
    'dataSize': 400,
    
    'iterations': 5000,
    'lr': 0.001,
    'lambda': 1
}

G = nx.complete_graph(config['nodeSize'])

data = np.random.normal(0, 1, config['dataSize'])

# byzantine = random.sample(range(config['nodeSize']), config['byzantineSize'])
# regular = list(set(range(config['nodeSize'])).difference(byzantine))
byzantine = [3]
regular = [0, 1, 2]
initial_point = np.random.normal(0, 1, config['nodeSize'])


def sign(a):
    if a > 0 :
        return 1
    elif a < 0:
        return -1
    else:
        return np.random.rand(1) * 2 - 1


class Agent():
    def __init__(self, id, para, workerPara, lr):
        self.id = id
        self.para = para
        self.workerPara = workerPara
        self.lr = lr

    def gradient(self, data):
        batchdata = random.sample(list(data), config['batchSize'])
        return np.mean(self.para - np.array(batchdata)) 

    def aggregate(self):
        penalty = 0
        for j in range(config['nodeSize']):
            if (self.id, j) in G.edges():
                penalty += sign(self.para - self.workerPara[j])
        return penalty

    def train(self, data):
        batch_gradient = self.gradient(data)
        penalty_gradient = self.aggregate()
        self.para -= self.lr * (batch_gradient +
                                config['lambda'] * penalty_gradient)

    def loss(self, data):
        result = 0.5 * np.linalg.norm(self.para - data) ** 2
        for j in regular:
            if (self.id, j) in G.edges():
                result += config['lambda'] / 2 * np.abs(self.para -
                                                        self.workerPara[j])
        return result
        # res = 0.5 * np.linalg.norm(data - self.para)**2
        # return res

    @property
    def get_data(self):
        return self.para


def compute_regular_data(data):
    num_data = config['dataSize'] // config['nodeSize']
    regular_data = []
    for i in range(config['dataSize']):
        for id in regular:
            if i in range(id * num_data, (id + 1) * num_data):
                regular_data.append(data[i])
                break
    return np.array(regular_data)


def averaged_para_loss(para, data):
    """
    Compute the loss of the averaged para
    """
    return 0.5 * np.linalg.norm(para - data) ** 2


def consensus_error(workerPara):
    return np.var(workerPara) 


def distance_to_optimal_para(workerPara, data):
    optimal_para = np.mean(data)
    return np.linalg.norm(workerPara - optimal_para) ** 2


def arbitray_attack(workerPara):
    for id in byzantine:
        workerPara[id] = np.random.randint(100) 
    return workerPara


def gaussian_attack(workerPara):
    for id in byzantine:
        workerPara[id] = np.random.normal(0, 100, size=1)
    return workerPara

def sign_flipping_attack(workerPara):
    for id in byzantine:
        workerPara[id] *= -5
    return workerPara


def get_learning(alpha, k):
    """
    Compute the decreasing learning step

    :param alpha: coefficient
    :param k: iteration time
    """
    return alpha / math.sqrt(k)



def main(attack, batchsize, lr, lamda, marker, ls):
    config['batchSize'] = batchsize
    config['lr'] = lr
    config['lambda'] = lamda
    num_data = config['dataSize'] // config['nodeSize']
    regular_data = compute_regular_data(data)
    # averaged_loss_list = []
    ce_list = []
    distance_list = []

    # labels = [
    #     'Batchsize = 1 (large variance)', 'Batchsize = 2 (medium variance)',
    #     'Batchsize = 3 (small variance)', 'Batchsize = 4 (zero variance)'
    # ]
    
    workerPara = initial_point.copy()
    print("The set of regular agents:", regular)
    print("The set of Byzantine agents:", byzantine)
    print("Start!")
    for k in tqdm(range(1, config['iterations'] + 1)):
        workerPara_memory = workerPara.copy()
        lr = config['lr']
        # lr = get_learning(config['lr'], k)

        if attack:
            workerPara_memory = attack(workerPara_memory)

        for id in regular:
            para = workerPara[id]
            model = Agent(id, para, workerPara_memory, lr)
            model.train(data[id * num_data:(id + 1) * num_data])
            workerPara[id] = model.get_data
        
        # averaged_loss = averaged_para_loss(np.mean(workerPara[regular]), regular_data)
        ce = consensus_error(workerPara[regular])
        distance = distance_to_optimal_para(workerPara[regular], regular_data)

        # print("The parameters of regular agents:{} ce:{} averaged_loss:{} distance:{}".format(
        #     workerPara[regular], ce, averaged_loss, distance))
        ce_list.append(ce)
        # averaged_loss_list.append(averaged_loss)
        distance_list.append(distance)

    label = 'batch size = ' + str(config['batchSize'])
    # plt.figure(1)
    # plt.plot(range(1, config['iterations'] + 1),
    #          ce_list, 
    #          label=label,
    #         #  marker=marker,
    #         #  markevery=500,
    #          ls=ls)
    # plt.ylabel('Consensus Error')
    # plt.xlabel('Iterations')
    # plt.legend()
    # plt.savefig('ToyExample_Ce.pdf')

    # plt.figure(2)
    # plt.plot(range(1, config['iterations'] + 1),
    #          averaged_loss_list,
    #          label=label)
    # plt.ylabel('Training Loss of Averaged Regular Model')
    # plt.xlabel('Iterations')
    # # plt.yscale('log')
    # plt.legend()
    # plt.savefig('ToyExample_Ave_Loss.pdf')
    
    plt.figure(3)
    plt.plot(range(1, config['iterations'] + 1),
             distance_list,
             label=label,
            #  marker=marker,
            #  markevery=500,
             ls=ls)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel(r'$||x^k - x^*||^2$', fontsize=15)
    plt.xlabel('Number of iterations', fontsize=15)
    # plt.yscale('log')
    plt.legend(fontsize=15)
    # plt.savefig('ToyExample_distance.pdf', bbox_inches='tight')


if __name__ == '__main__':
    batchsize_list = [1, 10, 50, 100]
    lr_list = [0.0001, 0.0002, 0.0003, 0.001]
    lamda_list = [0.005, 0.005, 0.005, 0.005]
    marker_list = ['o', 's', '^', 'v']
    ls_list = [':',  '-.', '--', '-']
    for batchsize, lr, lamda, marker, ls in zip(batchsize_list, lr_list, lamda_list, marker_list, ls_list):
        main(attack=gaussian_attack, batchsize=batchsize, lr=lr, lamda=lamda, marker=marker, ls=ls)
    plt.show()
