import os
import sys
os.chdir(sys.path[0])
sys.path.append("../../")
import numpy as np
import pickle
import random
from LoadMnist import getData, data_redistribute
from tqdm import tqdm
import Config
from MainModel import Softmax, get_accuracy, get_vars, get_learning
from Attacks import without_attacks, same_value_attacks, gaussian_attacks, \
    sign_flipping_attacks, sample_duplicating_attacks
import logging.config


logging.config.fileConfig(fname='..\\..\\Log\\loginit.ini', disable_existing_loggers=False)
logger = logging.getLogger("infoLogger")


class OursWorker(Softmax):

    def __init__(self, para, id, workerPara, config, lr, k):
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
        self.iter = k

    def lsvrg(self, image, label):
        # cal full grad in time 1
        if self.iter == 1:
            snapshot_fullGrad = self.cal_batch_grad(image, label)
            self.config['snapshot_fullGrad'][self.id] = snapshot_fullGrad
            return snapshot_fullGrad

        # cal full grad with prob 1/J
        pro = np.random.rand()
        if pro >= 1 / len(label):
            snapshot_para = self.config['snapshot_para'][self.id]
            snapshot_fullGrad = self.config['snapshot_fullGrad'][self.id]
        else:
            snapshot_para = self.para.copy()
            snapshot_fullGrad = self.cal_batch_grad(image, label)
            self.config['snapshot_para'][self.id] = snapshot_para
            self.config['snapshot_fullGrad'][self.id] = snapshot_fullGrad

        # cal stochastic grad
        select = np.random.randint (len(label))
        batchsize = self.config['batchSize']
        X = np.array (image[select : select + batchsize])
        Y = np.array (label[select : select + batchsize])
        Y = self.one_hot (Y)
        t = np.dot (self.para, X.T)
        t = t - np.max (t, axis=0)
        pro = np.exp (t) / np.sum (np.exp (t), axis=0)
        partial_gradient = - np.dot ((Y.T - pro), X) / batchsize + self.config['decayWeight'] * self.para

        # cal stochastic grad of snapshot
        t = np.dot (snapshot_para, X.T)
        t = t - np.max (t, axis=0)
        pro = np.exp (t) / np.sum (np.exp (t), axis=0)
        partial_gradient_snapshot = - np.dot ((Y.T - pro), X) / batchsize + self.config[
            'decayWeight'] * snapshot_para

        return partial_gradient - partial_gradient_snapshot + snapshot_fullGrad

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
        partial_gradient = self.lsvrg(image, label)
        aggregate_gradient = self.aggregate()
        self.para = self.para - self.lr * (aggregate_gradient + partial_gradient)


def ours(setting, attack, dataset, test_acc_flag, exp_lambda):
    """
    Run our proposed method in iid and non-iid settings under Byzantine attacks

    :param setting: 'iid' or 'noniid'
    :param attack: same-value attacks, sign-flipping attacks
                   sample-duplicating attacks(non-iid case)
    """
    print("The set of Byzantine agents:", Config.byzantine)
    print("The set of regular agents:", Config.regular)

    # Load the configurations
    conf = Config.DrsaLSVRGConfig.copy()
    num_data = int(Config.mnistConfig['trainNum'] / conf['nodeSize'])

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
    if not test_acc_flag :
        with open ("../../optimal-para/" + dataset + "-optimal-para-"+str(conf['penaltyPara'])+"-"+str(conf['byzantineSize'])+".pkl", 'rb') as f:
            para_star = pickle.load (f)
            print (para_star)

    # Rearrange the training data to simulate the non-i.i.d. case
    if setting == 'noniid':
        image_train, label_train = data_redistribute(image_train, label_train)

    # Parameter initialization
    workerPara = np.zeros((conf['nodeSize'], 10, 784))

    conf['snapshot_fullGrad'] = np.zeros((conf['nodeSize'], 10, 784))
    conf['snapshot_para'] = workerPara.copy()

    # Start training
    max_iteration = conf['iterations']
    select = random.choice(Config.regular)

    logger.info("Start!")
    for k in tqdm(range(1, max_iteration + 1)):
        count = 0
        workerPara_memory = workerPara.copy()
        lr = conf['learningStep']

        # Byzantine attacks
        workerPara_memory, last_str = attack(workerPara_memory)

        # x_i^{k+1} = x_i^k - lr * g_i^k
        for id in range(conf['nodeSize']):
            para = workerPara[id]
            model = OursWorker(para, id, workerPara_memory, conf, lr, k)
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

        # Test
        if test_acc_flag:
            # if k % 200 == 0 or k == 1:
                acc = get_accuracy(workerPara[select], image_test, label_test)
                acc_list.append(acc)
                var = get_vars(Config.regular, workerPara)
                var_list.append(var)
                logger.info ('the {}th iteration  test_acc:{} variance:{}'.format (k, acc, var))
        else:
            W_regular_norm = 0
            for i in Config.regular:
                W_regular_norm += np.linalg.norm(workerPara[i] - para_star) ** 2
            para_norm.append(W_regular_norm)
            # logger.info ('the {}th iteration para_norm: {}'.format(k, W_regular_norm))

    # Save the experiment results
    if test_acc_flag:
        if exp_lambda:
            output = open ("../../experiment-results-"+dataset+"/december-lsvrg" + last_str + "-lambda-" + str(conf['penaltyPara'])  + ".pkl", "wb")
            pickle.dump ((acc_list, var_list), output, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            output = open ("../../experiment-results-"+dataset+"/december-lsvrg" + last_str + "-" + str(conf['byzantineSize']) + ".pkl", "wb")
            pickle.dump((acc_list, var_list), output, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        output = open("../../experiment-results-"+dataset+"/december-lsvrg" + last_str + "-" + setting + "-para.pkl", "wb")
        pickle.dump(para_norm, output, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    ours(setting='noniid', attack=sample_duplicating_attacks, dataset='MNIST', test_acc_flag=True, exp_lambda=False)