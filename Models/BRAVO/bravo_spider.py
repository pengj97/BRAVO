import numpy as np
import pickle
import random
from LoadMnist import getData, data_redistribute
import Config
from MainModel import Softmax, get_accuracy, get_vars, get_learning
from Attacks import same_value_attack, sign_flipping_attacks, sample_duplicating_attack
import logging.config

logging.config.fileConfig (fname='..\\..\\Log\\loginit.ini', disable_existing_loggers=False)
logger = logging.getLogger ("infoLogger")


class OursWorker (Softmax) :

    def __init__(self, para, id, workerPara, config, lr, iter) :
        """
        Initialize the solver for regular workers

        :param para: model parameter, shape(10, 784)
        :param id: id of worker
        :param workerPara: the set of regular model parameters
        :param config: configuration
        :param lr: learning step
        """
        super ( ).__init__ (para, config)
        self.id = id
        self.workerPara = workerPara
        self.lr = lr
        self.iter = iter

    def aggregate(self) :
        """
        Aggregate the received models (consensus step)
        """
        penalty = np.zeros ((10, 784))
        for j in range (self.config['nodeSize']) :
            if (self.id, j) in Config.G.edges ( ) :
                penalty += np.sign (self.para - self.workerPara[j])

        aggregate_gradient = self.config['penaltyPara'] * penalty
        return aggregate_gradient

    def spider(self, image, label) :
        if self.iter % len(label) == 1:
            gradient = self.cal_batch_grad(image, label)
            self.config['snapshot_fullGrad'][self.id] = gradient.copy()
            return gradient
        else:
            # 计算随机梯度
            select = np.random.randint (len (label))
            batchsize = self.config['batchSize']
            X = np.array (image[select : select + batchsize])
            Y = np.array (label[select : select + batchsize])
            Y = self.one_hot (Y)
            t = np.dot (self.para, X.T)
            t = t - np.max (t, axis=0)
            pro = np.exp (t) / np.sum (np.exp (t), axis=0)
            partical_gradient = - np.dot ((Y.T - pro), X) / batchsize + self.config['decayWeight'] * self.para

            # 计算snapshot的随机梯度
            t = np.dot (self.config['iter_last'][self.id], X.T)
            t = t - np.max (t, axis=0)
            pro = np.exp (t) / np.sum (np.exp (t), axis=0)
            partical_gradient_snapshot = - np.dot ((Y.T - pro), X) / batchsize + self.config[
                'decayWeight'] * self.config['iter_last'][self.id]

            gradient = partical_gradient - partical_gradient_snapshot + self.config['snapshot_fullGrad'][self.id]
            return gradient

    def train(self, image, label) :
        """
        Updata the local model

        :param image: shape(10, 784)
        :param label: scalar
        :return:
        """
        partical_gradient = self.spider(image, label)
        aggregate_gradient = self.aggregate()
        gradient = partical_gradient + aggregate_gradient
        stepsize = self.config['learningStep'] * min(2 * self.config['epsilon'] / np.linalg.norm(gradient), 1)

        self.para = self.para - stepsize * gradient
        self.config['iter_last'][self.id] = self.para.copy()


def gragh_timevarying(G, pe) :
    """
    Generate the time-varing graph

    :param G: the topology graph
    :param pe: the probability of nodes connecting
    """
    tvG = G.copy ( )
    remove_edges = []
    all_edges = []
    for edge in tvG.edges ( ) :
        all_edges.append (edge)
        pro = np.random.rand ( )
        if pro >= pe :
            remove_edges.append (edge)
    # print(remove_edges)
    tvG.remove_edges_from (remove_edges)
    return tvG


def ours(setting, attack, flag_time_varying) :
    """
    Run our proposed method in iid and non-iid settings under Byzantine attacks

    :param setting: 'iid' or 'noniid'
    :param attack: same-value attacks, sign-flipping attacks
                   sample-duplicating attacks(non-iid case)
    :param flag_time_varying: whether the graph is time-varying
    """
    print (Config.byzantine)
    print (Config.regular)

    # Load the configurations
    conf = Config.DrsaSpiderConfig.copy ( )
    num_data = int (Config.mnistConfig['trainNum'] / conf['nodeSize'])

    loss_list = []
    para_list = []

    # Get the training data
    image_train, label_train = getData ('../../MNIST/train-images.idx3-ubyte',
                                        '../../MNIST/train-labels.idx1-ubyte')

    # Rearrange the training data to simulate the non-i.i.d. case
    if setting == 'noniid':
        image_train, label_train = data_redistribute (image_train, label_train)

    # Parameter initialization
    workerPara = np.zeros ((conf['nodeSize'], 10, 784))
    conf['iter_last'] = workerPara.copy()
    conf['snapshot_fullGrad'] = np.zeros((conf['nodeSize'], 10, 784))
    conf['epsilon'] = 1e2

    # Start training
    k = 0
    max_iteration = conf['iterations']
    last_str = '-wa'
    select = random.choice (Config.regular)

    logger.info ("Start!")
    while k < max_iteration :
        k += 1
        count = 0
        workerPara_memory = workerPara.copy ( )
        # lr = get_learning(conf['learningStep'], k)  # decreasing learning step
        lr = conf['learningStep']  # constant learning step

        # generate time-varying graph
        if flag_time_varying :
            graph_memory = Config.G.copy ( )
            Config.G = gragh_timevarying (graph_memory, pe=0.01)  # 生成时变图

        # Byzantine attacks
        if attack != None :
            workerPara_memory, last_str = attack (workerPara_memory)

        # Regular workers receive models from their neighbors
        # and update their local models
        for id in range (conf['nodeSize']) :
            para = workerPara[id]
            model = OursWorker (para, id, workerPara_memory, conf, lr, k)

            if setting == 'iid':
                model.train(image_train[id * num_data : (id + 1) * num_data],
                            label_train[id * num_data : (id + 1) * num_data])
                workerPara[id] = model.get_para()
                conf['iter_last'][id] = workerPara[id].copy()
            elif setting == 'noniid':
                if id in Config.regular:
                    model.train(image_train[count * num_data : (count + 1) * num_data],
                                label_train[count * num_data : (count + 1) * num_data])
                    workerPara[id] = model.get_para ()
                    conf['iter_last'][id] = workerPara[id].copy ()
                    count += 1

            if id == select :
                loss = model.cal_loss (image_train, label_train)
                loss_list.append (loss)
                para_list.append(para)

        # Testing
        logger.info ('the {}th iteration  loss:{}'.format (k, loss))

    # Save the experiment results
    output = open ("../../experiment-results/drsa-spider"+last_str+".pkl", "wb")
    pickle.dump ((para_list, loss_list), output, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__' :
    ours (setting='iid', attack=None, flag_time_varying=False)