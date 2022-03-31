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

    def __init__(self, para, id, workerPara, config, lr, optimizer, k):
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
        self.fun = OursWorker.__dict__[optimizer]
        self.iter = k

    def sgd(self, image, label):
        return self.cal_minibatch_sto_grad(image, label)

    def saga(self, image, label) :
        # 计算原始随机梯度
        select = np.random.randint (len (label))
        batchsize = self.config['batchSize']
        X = np.array (image[select : select + batchsize])
        Y = np.array (label[select : select + batchsize])
        Y = self.one_hot (Y)
        t = np.dot (self.para, X.T)
        t = t - np.max (t, axis=0)
        pro = np.exp (t) / np.sum (np.exp (t), axis=0)
        partial_gradient = - np.dot ((Y.T - pro), X) + self.config['decayWeight'] * self.para

        # 计算$\nabla f(x_i^k, \xi_i^k) - \nabla(\fi_i^k, \xi_i^k)$的差值
        if '{}'.format (select) in self.config['gradientTable'][self.id] :
            dvalue = partial_gradient - self.config['gradientTable'][self.id]['%d' % select]
        else :
            dvalue = partial_gradient

        # 更新梯度表
        self.config['gradientTable'][self.id]['{}'.format (select)] = partial_gradient

        # 利用SAGA方法得到梯度的估计值
        partial_gradient = dvalue + self.config['gradientEstimate'][self.id]

        # 更新$\bar{g}_i^k$
        self.config['gradientEstimate'][self.id] += dvalue / len (label)

        return partial_gradient

    def lsvrg(self, image, label):
        # 0时刻求全梯度
        if self.iter == 1:
            snapshot_fullGrad = self.cal_batch_grad(image, label)
            self.config['snapshot_fullGrad'][self.id] = snapshot_fullGrad

        # 依概率1/n取全梯度
        pro = np.random.rand()
        if pro >= 1 / len(label):
            snapshot_para = self.config['snapshot_para'][self.id]
            snapshot_fullGrad = self.config['snapshot_fullGrad'][self.id]
        else:
            snapshot_para = self.para.copy()
            snapshot_fullGrad = self.cal_batch_grad(image, label)
            self.config['snapshot_para'][self.id] = snapshot_para
            self.config['snapshot_fullGrad'][self.id] = snapshot_fullGrad

        # 计算随机梯度
        select = np.random.randint (len(label))
        batchsize = self.config['batchSize']
        X = np.array (image[select : select + batchsize])
        Y = np.array (label[select : select + batchsize])
        Y = self.one_hot (Y)
        t = np.dot (self.para, X.T)
        t = t - np.max (t, axis=0)
        pro = np.exp (t) / np.sum (np.exp (t), axis=0)
        partial_gradient = - np.dot ((Y.T - pro), X) / batchsize + self.config['decayWeight'] * self.para

        # 计算snapshot的随机梯度
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
        partial_gradient = self.fun (self, image, label)
        aggregate_gradient = self.aggregate()
        self.para = self.para - self.lr * (aggregate_gradient + partial_gradient)


def ours(setting, attack, optimizer):
    """
    Run our proposed method in iid and non-iid settings under Byzantine attacks

    :param setting: 'iid' or 'noniid'
    :param attack: same-value attacks, sign-flipping attacks
                   sample-duplicating attacks(non-iid case)
    :param optimizer: 'sgd' or 'lsvrg' or 'saga'
    """
    print("The set of Byzantine agents:", Config.byzantine)
    print("The set of regular agents:", Config.regular)

    # Load the configurations
    conf = Config.DrsaConfig.copy()
    num_data = int(Config.mnistConfig['trainNum'] / conf['nodeSize'])

    loss_list = []
    para_list = []

    # Get the training data
    image_train, label_train = getData('../../MNIST/train-images.idx3-ubyte',
                                       '../../MNIST/train-labels.idx1-ubyte')

    # Get the testing data
    image_test, label_test = getData ('../../MNIST/t10k-images.idx3-ubyte',
                                      '../../MNIST/t10k-labels.idx1-ubyte')

    # Rearrange the training data to simulate the non-i.i.d. case
    if setting == 'noniid':
        image_train, label_train = data_redistribute(image_train, label_train)

    # Parameter initialization
    workerPara = np.zeros((conf['nodeSize'], 10, 784))
    conf['gradientTable'] = [{} for _ in range(conf['nodeSize'])]
    conf['gradientEstimate'] = np.zeros((conf['nodeSize'], 10, 784))
    conf['snapshot_fullGrad'] = np.zeros((conf['nodeSize'], 10, 784))
    conf['snapshot_para'] = workerPara.copy()
    conf['iter_last'] = workerPara.copy()

    # Start training
    k = 0
    max_iteration = conf['iterations']
    last_str = '-wa'
    select = random.choice(Config.regular)

    logger.info("Start!")
    while k <= max_iteration:
        k += 1
        count = 0
        workerPara_memory = workerPara.copy()
        if optimizer == 'sgd':
            lr = get_learning(conf['learningStep'], k)
        else:
            lr = conf['learningStep']

        # Byzantine attacks
        if attack != None:
            workerPara_memory, last_str = attack(workerPara_memory)

        # x_i^{k+1} = x_i^k - lr * g_i^k
        for id in range(conf['nodeSize']):
            para = workerPara[id]
            model = OursWorker(para, id, workerPara_memory, conf, lr, optimizer, k)
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
            if k % 1000 == 1:
                if id == select:
                    acc = get_accuracy(workerPara[id], image_test, label_test)
                    print('test acc:', acc)
                    loss = model.cal_loss(image_train, label_train)
                    loss_list.append(loss)
                    para_list.append(para)
                    logger.info ('the {}th iteration  loss:{}'.format (k, loss))

    # Save the experiment results
    output = open("../../experiment-results/drsa-"+optimizer+"-old"+last_str+"-"+setting+".pkl", "wb")
    pickle.dump((para_list, loss_list), output, protocol=pickle.HIGHEST_PROTOCOL)
    print(optimizer)


if __name__ == '__main__':
    ours(setting='noniid', attack=sample_duplicating_attack, optimizer='lsvrg')