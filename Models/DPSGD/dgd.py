import numpy as np
import pickle
import random
from LoadMnist import getData, data_redistribute
import Config
from MainModel import Softmax, get_accuracy, get_vars, get_learning
from Attacks import without_attacks, gaussian_attacks, alie_attacks, \
    same_value_attacks, sign_flipping_attacks, sample_duplicating_attacks
import logging.config

logging.config.fileConfig (fname='..\\..\\Log\\loginit.ini', disable_existing_loggers=False)
logger = logging.getLogger ("infoLogger")


class DGDWorker (Softmax) :

    def __init__(self, para, id, workerPara, config, lr) :
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

    def aggregate(self) :
        """
        Aggregate the received models (consensus step)
        """
        aggregate_para = np.zeros_like (self.para)
        for j in range (self.config['nodeSize']) :
            aggregate_para += Config.weight_matrix[self.id][j] * self.workerPara[j]
        return aggregate_para

    def train(self, image, label) :
        """
        Updata the local model

        :param image: shape(10, 784)
        :param label: scalar
        :return:
        """
        partical_gradient = self.cal_batch_grad(image, label)
        aggregate_para = self.aggregate()
        self.para = aggregate_para - self.lr * partical_gradient


def dpsgd(setting, attack) :
    """
    Run DPSGD method in iid and non-iid settings under Byzantine attacks

    :param setting: 'iid' or 'noniid'
    :param attack: same-value attacks, sign-flipping attacks
                   sample-duplicating attacks(non-iid case)
    """
    print (Config.byzantine)
    print (Config.regular)

    # Load the configurations
    conf = Config.DPSGDConfig.copy ( )
    num_data = int (Config.mnistConfig['trainNum'] / conf['nodeSize'])

    classification_accuracy = []
    variances = []

    # Get the training data
    image_train, label_train = getData ('../../MNIST/train-images.idx3-ubyte',
                                        '../../MNIST/train-labels.idx1-ubyte')

    # Rearrange the training data to simulate the non-i.i.d. case
    if setting == 'noniid' :
        image_train, label_train = data_redistribute (image_train, label_train)

    # Get the testing data
    image_test, label_test = getData ('../../MNIST/t10k-images.idx3-ubyte',
                                      '../../MNIST/t10k-labels.idx1-ubyte')

    # Parameter initialization
    workerPara = np.zeros ((conf['nodeSize'], 10, 784))

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

        lr = get_learning (conf['learningStep'], k)  # compute decreasing learning rate

        # Byzantine attacks
        if attack != None :
            workerPara_memory, last_str = attack (workerPara_memory)

        # Regular workers receive models from their neighbors
        # and update their local models
        for id in range (conf['nodeSize']) :
            para = workerPara[id]
            model = DGDWorker (para, id, workerPara_memory, conf, lr)
            if setting == 'iid' :
                model.train (image_train[id * num_data : (id + 1) * num_data],
                             label_train[id * num_data : (id + 1) * num_data])
                workerPara[id] = model.get_para
            elif setting == 'noniid' :
                if id in Config.regular :
                    model.train (image_train[count * num_data : (count + 1) * num_data],
                                 label_train[count * num_data : (count + 1) * num_data])
                    workerPara[id] = model.get_para
                    count += 1

        logger.info ('the {}th iteration'.format (k))

    # Save the experiment results
    output = open ("../../experiment-results/dgd-optimal-para-" + str(conf['byzantineSize']) + '-' + setting + ".pkl", "wb")
    pickle.dump (workerPara[select], output, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__' :
    dpsgd (setting='iid', attack=without_attacks)
