import os
import sys
os.chdir(sys.path[0])
sys.path.append("../../")
import numpy as np
import pickle
import random
from tqdm import tqdm
from LoadMnist import getData, data_redistribute
import Config
from MainModel import Softmax, get_accuracy, get_vars, get_learning, get_learning_v2
from Attacks import without_attacks, same_value_attacks, \
    sign_flipping_attacks, sample_duplicating_attacks, gaussian_attacks
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

    def sgd(self, image, label):
        return self.cal_minibatch_sto_grad(image, label)

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

    def train(self, image, label):
        partial_gradient = self.sgd(image, label)
        aggregate_gradient = self.aggregate()
        self.para = self.para - self.lr * (aggregate_gradient + partial_gradient)


def ours(setting, attack, dataset, test_acc_flag, stepsize_experiment_flag):
    """
    Run our proposed method in iid and non-iid settings under Byzantine attacks

    :param setting: 'iid' or 'noniid'
    :param attack: same-value attacks, sign-flipping attacks
                   sample-duplicating attacks(non-iid case)
    """
    print("The set of Byzantine agents:", Config.byzantine)
    print("The set of regular agents:", Config.regular)

    # Load the configurations
    conf = Config.DrsaConfig.copy()
    num_data = int(Config.mnistConfig['trainNum'] / conf['nodeSize'])

    loss_list = []
    acc_list = []
    var_list = []
    para_norm = []

    # Get the training data
    image_train, label_train = getData('../../' + dataset + '/train-images.idx3-ubyte',
                                       '../../' + dataset + '/train-labels.idx1-ubyte')

    # Get the testing data
    image_test, label_test = getData ('../../' + dataset + '/t10k-images.idx3-ubyte',
                                      '../../' + dataset + '/t10k-labels.idx1-ubyte')
    # Get the optimal solution
    if dataset == 'MNIST':
        if not test_acc_flag :
            with open ("../../optimal-para/optimal-para-"+str(conf['penaltyPara'])+"-"+str(conf['byzantineSize'])+".pkl", 'rb') as f:
                para_star = pickle.load (f)
                print (para_star)
    elif dataset == 'FashionMNIST':
        if not test_acc_flag :
            with open ("../../experiment-results/" + dataset + "-optimal-para-"+str(conf['penaltyPara'])+"-"+str(conf['byzantineSize'])+".pkl", 'rb') as f:
                para_star = pickle.load (f)
                print (para_star)

    # Rearrange the training data to simulate the non-i.i.d. case
    if setting == 'noniid':
        image_train, label_train = data_redistribute(image_train, label_train)

    # Parameter initialization
    # workerPara = np.random.random((conf['nodeSize'], 10, 784))
    workerPara = np.zeros((conf['nodeSize'], 10, 784))

    # Start training
    max_iteration = conf['iterations']
    select = random.choice(Config.regular)

    logger.info("Start!")
    for k in tqdm(range(1, max_iteration + 1)):
        count = 0
        workerPara_memory = workerPara.copy()
        # 0: \alpha / sqrt{k}
        # 1: \alpha / k
        # 2: \alpha 
        if stepsize_experiment_flag == 0:
            lr = get_learning(conf['learningStep'], k)
        elif stepsize_experiment_flag == 1:
            lr = get_learning_v2(conf['learningStep'], k)
        elif stepsize_experiment_flag == 2:
            lr = conf['learningStep']

        # Byzantine attacks
        workerPara_memory, last_str = attack(workerPara_memory)

        # x_i^{k+1} = x_i^k - lr * g_i^k
        for id in range(conf['nodeSize']):
            para = workerPara[id]
            model = OursWorker(para, id, workerPara_memory, conf, lr)
            if setting == 'iid':
                model.train(image_train[id * num_data: (id + 1) * num_data],
                            label_train[id * num_data: (id + 1) * num_data])
                workerPara[id] = model.get_para
            elif setting == 'noniid':
                if id in Config.regular:
                    model.train(image_train[count * num_data : (count + 1) * num_data],
                                label_train[count * num_data : (count + 1) * num_data])
                    workerPara[id] = model.get_para
                    count += 1

        # Testing
        if test_acc_flag :
            # if k % 200 == 0 or k == 1 :
                acc = get_accuracy (workerPara[select], image_test, label_test)
                acc_list.append (acc)
                var = get_vars (Config.regular, workerPara)
                var_list.append (var)
                logger.info ('the {}th iteration  test_acc:{} variance:{}'.format (k, acc, var))

        else :
            W_regular_norm = 0
            for i in Config.regular :
                W_regular_norm += np.linalg.norm (workerPara[i] - para_star) ** 2
            para_norm.append (W_regular_norm)
            # logger.info ('the {}th iteration para_norm: {}'.format (k, W_regular_norm))

    # Save the experiment results
    if test_acc_flag :
        output = open ("../../experiment-results-"+dataset+"/december" + last_str + "-" + str(conf['byzantineSize']) + ".pkl", "wb")
        pickle.dump ((acc_list, var_list), output, protocol=pickle.HIGHEST_PROTOCOL)
    else :
        if stepsize_experiment_flag == 0:
            output = open ("../../experiment-results-"+dataset+"-2/december-sqrt" + last_str + "-" + setting + "-para.pkl", "wb")
            pickle.dump (para_norm, output, protocol=pickle.HIGHEST_PROTOCOL)
        elif stepsize_experiment_flag == 1:
            output = open ("../../experiment-results-"+dataset+"-2/december" + last_str + "-" + setting + "-para.pkl", "wb")
            pickle.dump (para_norm, output, protocol=pickle.HIGHEST_PROTOCOL)
        elif stepsize_experiment_flag == 2:
            output = open ("../../experiment-results-"+dataset+"-2/december-constant" + last_str + "-" + setting + "-para.pkl", "wb")
            pickle.dump (para_norm, output, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    ours(setting='iid', attack=sign_flipping_attacks, dataset='FashionMNIST', test_acc_flag=True, stepsize_experiment_flag=0)