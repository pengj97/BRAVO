import numpy as np
import pickle
import random
from LoadMnist import getData, data_redistribute
import Config
from MainModel import Softmax, get_accuracy, get_vars, get_learning
from Attacks import same_value_attack, sign_flipping_attacks, sample_duplicating_attack
import logging.config

logging.config.fileConfig(fname='..\\..\\Log\\loginit.ini', disable_existing_loggers=False)
logger = logging.getLogger("infoLogger")


class OursWorker(Softmax):

    def __init__(self, para, id, workerPara, config, lr):
        """
        Initialize the solver for regular workers

        :param para: model parameter, shape(10, 784)
        :param id: id of worker
        :param workerPara: the set of regular model parameters
        :param config: configuration
        :param lr: learning step
        """
        super().__init__(para, config)
        self.id = id
        self.workerPara = workerPara
        self.lr = lr

    def aggregate(self):
        """
        Aggregate the received models (consensus step)
        """
        penalty = np.zeros((10, 784))
        for j in range(self.config['nodeSize']):
            if (self.id, j) in Config.G.edges():
                penalty += np.sign(self.para - self.workerPara[j])

        aggregate_gradient = self.config['penaltyPara'] * penalty
        return aggregate_gradient

    def saga(self, image, label) :
        # 计算原始随机梯度
        select = np.random.randint(len(label))
        batchsize = self.config['batchSize']
        X = np.array (image[select : select + batchsize])
        Y = np.array (label[select : select + batchsize])
        Y = self.one_hot(Y)
        t = np.dot(self.para, X.T)
        t = t - np.max(t, axis=0)
        pro = np.exp(t) / np.sum(np.exp(t), axis=0)
        partical_gradient = - np.dot((Y.T - pro), X) + self.config['decayWeight'] * self.para

        # 计算$\nabla f(x_i^k, \xi_i^k) - \nabla(\fi_i^k, \xi_i^k)$的差值
        if '{}'.format (select) in self.config['gradientTable'][self.id] :
            dvalue = partical_gradient - self.config['gradientTable'][self.id]['%d' % select]
        else :
            dvalue = partical_gradient

        # 更新梯度表
        self.config['gradientTable'][self.id]['{}'.format (select)] = partical_gradient

        # 利用SAGA方法得到梯度的估计值
        partical_gradient = dvalue + self.config['gradientEstimate'][self.id]

        # 更新$\bar{g}_i^k$
        self.config['gradientEstimate'][self.id] += dvalue / len(label)

        return partical_gradient

    def train(self, image, label):
        """
        Updata the local model

        :param image: shape(10, 784)
        :param label: scalar
        :return:
        """
        partical_gradient = self.saga(image, label)  # saga
        aggregate_gradient = self.aggregate()

        self.para = self.para - self.lr * (partical_gradient + aggregate_gradient)


def gragh_timevarying(G, pe):
    """
    Generate the time-varing graph

    :param G: the topology graph
    :param pe: the probability of nodes connecting
    """
    tvG = G.copy()
    remove_edges = []
    all_edges = []
    for edge in tvG.edges():
        all_edges.append(edge)
        pro = np.random.rand()
        if pro >= pe:
            remove_edges.append(edge)
    # print(remove_edges)
    tvG.remove_edges_from(remove_edges)
    return tvG


def ours(setting, attack, flag_time_varying):
    """
    Run our proposed method in iid and non-iid settings under Byzantine attacks

    :param setting: 'iid' or 'noniid'
    :param attack: same-value attacks, sign-flipping attacks
                   sample-duplicating attacks(non-iid case)
    :param flag_time_varying: whether the graph is time-varying
    """
    print(Config.byzantine)
    print(Config.regular)

    # Load the configurations
    conf = Config.DrsaSAGAConfig.copy()
    num_data = int(Config.mnistConfig['trainNum'] / conf['nodeSize'])

    loss_list = []
    para_list = []

    # Get the training data
    image_train, label_train = getData('../../MNIST/train-images.idx3-ubyte',
                                       '../../MNIST/train-labels.idx1-ubyte')

    # Rearrange the training data to simulate the non-i.i.d. case
    if setting == 'noniid':
        image_train, label_train = data_redistribute(image_train, label_train)

    # Parameter initialization
    workerPara = np.zeros((conf['nodeSize'], 10, 784))
    conf['gradientTable'] = [{} for _ in range(conf['nodeSize'])]
    conf['gradientEstimate'] = np.zeros((conf['nodeSize'], 10, 784))

    # Start training
    k = 0
    max_iteration = conf['iterations']
    last_str = '-wa'
    select = random.choice(Config.regular)

    logger.info("Start!")
    while k < max_iteration:
        k += 1
        count = 0
        workerPara_memory = workerPara.copy()
        lr = conf['learningStep']  # constant learning step

        # generate time-varying graph
        if flag_time_varying:
            graph_memory = Config.G.copy()
            Config.G = gragh_timevarying(graph_memory, pe=0.01)  # 生成时变图

        # Byzantine attacks
        if attack != None:
            workerPara_memory, last_str = attack(workerPara_memory)

        # Regular workers receive models from their neighbors
        # and update their local models
        for id in range(conf['nodeSize']):
            para = workerPara[id]
            model = OursWorker(para, id, workerPara_memory, conf, lr)
            if setting == 'iid':
                model.train(image_train[id * num_data: (id + 1) * num_data],
                            label_train[id * num_data: (id + 1) * num_data])
                workerPara[id] = model.get_para()
            elif setting == 'noniid':
                if id in Config.regular:
                    model.train (image_train[count * num_data : (count + 1) * num_data],
                                 label_train[count * num_data : (count + 1) * num_data])
                    workerPara[id] = model.get_para()
                    count += 1

            if id == select:
                loss = model.cal_loss(image_train, label_train)
                loss_list.append(loss)
                para_list.append(para)

        # Testing
        logger.info ('the {}th iteration  loss:{}'.format(k, loss))

    # Save the experiment results
    output = open("../../experiment-results/drsa-saga"+last_str+".pkl", "wb")
    pickle.dump((para_list, loss_list), output, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    ours(setting='iid', attack=None, flag_time_varying=False)