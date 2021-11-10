import numpy as np
import pickle
import copy
import random
from LoadMnist import getData
import Config
from MainModel import Softmax, get_accuracy, get_vars, get_learning
import logging.config


logging.config.fileConfig(fname='..\\..\\Log\\loginit.ini', disable_existing_loggers=False)
logger = logging.getLogger("infoLogger")


class GdSoftmax(Softmax):

    def __init__(self, para, config, lr, optimizer, k):
        super().__init__(para, config)
        self.lr = lr
        self.fun = GdSoftmax.__dict__[optimizer]
        self.iter = k

    def sgd(self, image, label):
        return self.cal_minibatch_sto_grad(image, label)

    def gd(self, image, label):
        return self.cal_batch_grad(image, label)

    def saga(self, image, label) :
        # 计算原始随机梯度
        select = np.random.randint(len (label))
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

        # 利用SAGA方法得到梯度的估计值
        gradient = dvalue + self.config['gradientEstimate']

        # 更新梯度表
        self.config['gradientTable']['{}'.format (select)] = partial_gradient

        # 更新$\bar{g}_i^k$
        self.config['gradientEstimate'] += dvalue / len(label)

        return gradient

    def lsvrg(self, image, label) :
        # 0时刻求全梯度
        if self.iter == 1 :
            snapshot_fullGrad = self.cal_batch_grad (image, label)
            self.config['snapshot_fullGrad'] = snapshot_fullGrad

        # 依概率1/n取全梯度
        pro = np.random.rand ( )
        if pro >= 1 / len (label) :
            snapshot_para = self.config['snapshot_para']
            snapshot_fullGrad = self.config['snapshot_fullGrad']
        else :
            snapshot_para = copy.deepcopy(self.para)
            snapshot_fullGrad = self.cal_batch_grad (image, label)
            self.config['snapshot_para']= snapshot_para
            self.config['snapshot_fullGrad']= snapshot_fullGrad

        # 计算随机梯度
        select = np.random.randint (len (label))
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

    def train(self, image, label):
        if self.iter % 1000 == 1:
            variance = np.linalg.norm(self.fun(self, image, label) - self.cal_batch_grad(image, label)) ** 2
            print("variance:", variance)
        gradient = self.fun(self, image, label)
        self.para = self.para - self.lr * gradient


def gd(optimizer):
    """
    Run GD method in single machine towards softmax regression problem
    """

    # Load the configurations
    conf = Config.GDConfig.copy()

    # Get the training data
    image_train, label_train = getData('../../MNIST/train-images.idx3-ubyte',
                                       '../../MNIST/train-labels.idx1-ubyte')

    # Parameter initialization
    para = np.zeros((10, 784))
    conf['gradientTable'] = {}
    conf['gradientEstimate'] = np.zeros((10, 784))
    conf['snapshot_fullGrad'] = np.zeros((10, 784))
    conf['snapshot_para'] = copy.deepcopy(para)

    loss = 0
    loss_list = []

    # Start training
    max_iteration = conf['iterations']

    logger.info("Start!")
    for k in range(1, max_iteration+2):
        lr = conf['learningStep']

        # Training
        model = GdSoftmax(para, conf, lr, optimizer, k)
        model.train(image_train, label_train)
        para = model.get_para

        logger.info("the %dth iteration" %k)
        # Testing
        # if k % 1000 == 1:
        #     loss = model.cal_loss(image_train, label_train)
        #     logger.info('the {}th iteration loss:{}'.format(k, loss))
        #     loss_list.append(loss)
        #     para_list.append(para)
    print(optimizer)
    print(para)

    output = open('../../optimal-para.pkl', 'wb')
    pickle.dump(para, output, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    gd(optimizer='gd')
