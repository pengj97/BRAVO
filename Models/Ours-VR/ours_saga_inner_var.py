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
from Attacks import without_attacks, same_value_attacks, \
    sign_flipping_attacks, sample_duplicating_attacks, gaussian_attacks
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

    def saga(self, image, label) :
        full_grad = self.cal_batch_grad(image, label)
        partial_gradient = np.zeros_like(full_grad)
        if self.iter == 1:
            for i in range(len(label)):
                X = np.array (image[i: i + 1])
                Y = np.array (label[i: i + 1])
                Y = self.one_hot (Y)
                t = np.dot (self.para, X.T)
                t = t - np.max (t, axis=0)
                pro = np.exp (t) / np.sum (np.exp (t), axis=0)
                partial_g = - np.dot ((Y.T - pro), X) + self.config['decayWeight'] * self.para
                self.config['gradientTable'][self.id]['%d' % i] = partial_g.copy()
            self.config['gradientEstimate'][self.id] = full_grad.copy()
        
        else:
            # 计算原始随机梯度
            select = np.random.randint (len (label))
            # batchsize = self.config['batchSize']
            batchsize = 1
            X = np.array (image[select : select + batchsize])
            Y = np.array (label[select : select + batchsize])
            Y = self.one_hot (Y)
            t = np.dot (self.para, X.T)
            t = t - np.max (t, axis=0)
            pro = np.exp (t) / np.sum (np.exp (t), axis=0)
            partial_gradient = - np.dot ((Y.T - pro), X) / batchsize + self.config['decayWeight'] * self.para

            # 计算$\nabla f(x_i^k, \xi_i^k) - \nabla f(\fi_i^k, \xi_i^k)$的差值
            if '{}'.format (select) in self.config['gradientTable'][self.id] :
                dvalue = partial_gradient - self.config['gradientTable'][self.id]['%d' % select]
            else :
                dvalue = partial_gradient

            # 更新梯度表
            self.config['gradientTable'][self.id]['{}'.format (select)] = partial_gradient.copy()

            # 利用SAGA方法得到梯度的估计值
            partial_gradient = dvalue + self.config['gradientEstimate'][self.id]

            # 更新$\bar{g}_i^k$
            self.config['gradientEstimate'][self.id] += dvalue / len (label)

        inner_var = 0
        for i in range(len(label)):
            X = np.array (image[i : i + 1])
            Y = np.array (label[i : i + 1])
            Y = self.one_hot (Y)
            t = np.dot (self.para, X.T)
            t = t - np.max (t, axis=0)
            pro = np.exp (t) / np.sum (np.exp (t), axis=0)
            partial_grad = - np.dot ((Y.T - pro), X) + self.config['decayWeight'] * self.para

            # 计算$\nabla f(x_i^k, \xi_i^k) - \nabla(\fi_i^k, \xi_i^k)$的差值
            if '{}'.format (i) in self.config['gradientTable'][self.id] :
                dvalue = partial_grad - self.config['gradientTable'][self.id]['%d' % i]
            else :
                dvalue = partial_grad

            # 利用SAGA方法得到梯度的估计值
            partial_grad = dvalue + self.config['gradientEstimate'][self.id]
            inner_var += np.linalg.norm(partial_grad - full_grad) ** 2 / len(label)
    
        return inner_var, partial_gradient

    def cal_inner_var_vr(self, image, label):
        inner_var = 0
        full_grad = self.cal_batch_grad(image, label)
        for i in range(len(label)):
            X = np.array (image[i : i + 1])
            Y = np.array (label[i : i + 1])
            Y = self.one_hot (Y)
            t = np.dot (self.para, X.T)
            t = t - np.max (t, axis=0)
            pro = np.exp (t) / np.sum (np.exp (t), axis=0)
            partial_grad = - np.dot ((Y.T - pro), X) + self.config['decayWeight'] * self.para

            # 计算$\nabla f(x_i^k, \xi_i^k) - \nabla(\fi_i^k, \xi_i^k)$的差值
            if '{}'.format (i) in self.config['gradientTable'][self.id] :
                dvalue = partial_grad - self.config['gradientTable'][self.id]['%d' % i]
            else :
                dvalue = partial_grad

            # 利用SAGA方法得到梯度的估计值
            partial_grad = dvalue + self.config['gradientEstimate'][self.id]
            inner_var += np.linalg.norm(partial_grad - full_grad) ** 2 / len(label)
        return inner_var

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
        inner_var, partial_gradient = self.saga(image, label)
        aggregate_gradient = self.aggregate()
        self.para = self.para - self.lr * (aggregate_gradient + partial_gradient)
        return inner_var


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
    conf = Config.DrsaSAGAConfig.copy()
    num_data = int(Config.mnistConfig['trainNum'] / conf['nodeSize'])

    loss_list = []
    acc_list = []
    var_list = []
    inner_var_list = []
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
    workerPara = np.zeros((conf['nodeSize'], 10, 784))

    conf['gradientTable'] = [{} for _ in range(conf['nodeSize'])]
    conf['gradientEstimate'] = np.zeros((conf['nodeSize'], 10, 784))

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

        inner_var = 0

        # x_i^{k+1} = x_i^k - lr * g_i^k
        for id in range(conf['nodeSize']):
            para = workerPara[id]
            model = OursWorker(para, id, workerPara_memory, conf, lr, k)
            if setting == 'iid':
                inner_var_agent = model.train(image_train[id * num_data: (id + 1) * num_data],
                                              label_train[id * num_data: (id + 1) * num_data])
                workerPara[id] = model.get_para
                # if id in Config.regular:
                #     inner_var_agent = model.cal_inner_variation(image_train[id * num_data: (id + 1) * num_data],
                #                                                 label_train[id * num_data: (id + 1) * num_data])
                #     inner_var = max(inner_var, inner_var_agent)
            elif setting == 'noniid':
                if id in Config.regular:
                    inner_var_agent = model.train(image_train[count * num_data : (count + 1) * num_data],
                                                  label_train[count * num_data : (count + 1) * num_data])
                    workerPara[id] = model.get_para
                    count += 1
            if id in Config.regular:
                inner_var += inner_var_agent / len(Config.regular)

        if test_acc_flag :
            inner_var_list.append(inner_var)
            logger.info ('the {}th iteration inner_var:{}'.format (k, inner_var))

        else :
            W_regular_norm = 0
            for i in Config.regular :
                W_regular_norm += np.linalg.norm (workerPara[i] - para_star) ** 2
            para_norm.append (W_regular_norm)

    # output = open ("../../experiment-results-"+dataset+"-2/december-saga" + last_str + "-" + str(conf['learningStep']) + "-inner-var-new.pkl", "wb")
    # pickle.dump (inner_var_list, output, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    ours(setting='iid', attack=without_attacks, dataset='MNIST', test_acc_flag=True, exp_lambda=False)
