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
from MainModel import Softmax, get_accuracy, get_vars, get_learning
from Attacks import without_attacks, gaussian_attacks, alie_attacks, \
    same_value_attacks, sign_flipping_attacks, sample_duplicating_attacks
import logging.config


logging.config.fileConfig(fname='..\\..\\Log\\loginit.ini', disable_existing_loggers=False)
logger = logging.getLogger("infoLogger")


class DPSGDWorker(Softmax):

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

    # def aggregate(self):
    #     """
    #     Aggregate the received models (consensus step)
    #     """
    #     neighbors_para = []
    #     for j in range(self.config['nodeSize']):
    #         if (self.id, j) in Config.G.edges() or self.id == j:
    #             neighbors_para.append(self.workerPara[j])
    #     aggregate_para = np.mean(neighbors_para, axis=0)
    #     return aggregate_para

    def aggregate(self):
        """
        Aggregate the received models (consensus step)
        """
        aggregate_para = np.zeros_like(self.para)
        for j in range(self.config['nodeSize']):
            aggregate_para += Config.weight_matrix[self.id][j] * self.workerPara[j]
        return aggregate_para

    def train(self, image, label):
        """
        Updata the local model

        :param image: shape(10, 784)
        :param label: scalar
        :return:
        """
        partical_gradient = self.cal_minibatch_sto_grad(image, label)
        aggregate_para = self.aggregate()
        self.para = aggregate_para - self.lr * partical_gradient


def dpsgd(setting, attack, dataset):
    """
    Run DPSGD method in iid and non-iid settings under Byzantine attacks

    :param setting: 'iid' or 'noniid'
    :param attack: same-value attacks, sign-flipping attacks
                   sample-duplicating attacks(non-iid case)
    """
    print("The set of Byzantine agents:", Config.byzantine)
    print("The set of regular agents:", Config.regular)

    # Load the configurations
    conf = Config.DPSGDConfig.copy()
    num_data = int(Config.mnistConfig['trainNum'] / conf['nodeSize'])

    classification_accuracy = []
    variances = []

    # Get the training data
    image_train, label_train = getData('../../' + dataset + '/train-images.idx3-ubyte',
                                       '../../' + dataset + '/train-labels.idx1-ubyte')

    # Rearrange the training data to simulate the non-i.i.d. case
    if setting == 'noniid':
        image_train, label_train = data_redistribute(image_train, label_train)

    # Get the testing data
    image_test, label_test = getData ('../../' + dataset + '/t10k-images.idx3-ubyte',
                                      '../../' + dataset + '/t10k-labels.idx1-ubyte')

    # Parameter initialization
    workerPara = np.random.random((conf['nodeSize'], 10, 784))

    # Start training
    k = 0
    max_iteration = conf['iterations']
    last_str = '-wa'
    select = random.choice(Config.regular)

    logger.info("Start!")
    for k in tqdm(range(1, max_iteration + 1)):
        count = 0
        workerPara_memory = workerPara.copy()

        # compute decreasing learning rate
        lr = get_learning(conf['learningStep'], k)

        # Byzantine attacks
        if attack != None:
            workerPara_memory, last_str = attack(workerPara_memory)

        # Regular workers receive models from their neighbors
        # and update their local models
        for id in range(conf['nodeSize']):
            para = workerPara[id]
            model = DPSGDWorker(para, id, workerPara_memory, conf, lr)
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
        if k % 200 == 0 or k == 1:
            acc = get_accuracy(workerPara[select], image_test, label_test)
            classification_accuracy.append(acc)
            var = get_vars(Config.regular, workerPara)
            variances.append(var)
            logger.info('the {}th iteration acc: {}, vars: {}'.format(k, acc, var))

    # print(classification_accuracy)
    # print(variances)

    # Save the experiment results
    output = open("../../experiment-results-"+dataset+"/dpsgd" + last_str + "-" + str(conf['byzantineSize']) + ".pkl", "wb")
    pickle.dump((classification_accuracy, variances), output, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    dpsgd(setting='iid', attack=gaussian_attacks, dataset='FashionMNIST')
