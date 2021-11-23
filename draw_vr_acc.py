import numpy as np
import matplotlib.pyplot as plt
import pickle
import Config

set_iteration_interval = []
for k in range(1, Config.optConfig['iterations']+1):
    if k % 200 == 0 or k == 1:
        set_iteration_interval.append(k)
colors = ['black',  'skyblue', 'blue', 'purple', 'green', 'red']
markers = ['h', 'v',  'x', '+', 's', 'o']


def draw(attack):
    """
    Draw the curve of experimental results

    :param attack:
                   'wa': without Byzantine attacks,
                   'sv': same-value attacks
                   'sf': sign-flipping attacks
                   'sd': sample-duplicating attacks in non-i.i.d. case
    """

    methods = ['dpsgd', 'byrdie_100_v4', 'bridge', 'december', 'december-saga', 'december-lsvrg']
    labels = ['DPSGD', 'ByRDiE-S', 'BRIDGE-S',  'DECEMBER', 'DECEMBER-SAGA', 'DECEMBER-LSVRG']

    acc_list = []
    var_list = []

    for i in range(len(methods)):
        with open("experiment-results-2/" + methods[i] + "-" + attack + ".pkl", "rb") as f:
            acc, var = pickle.load(f)
            acc_list.append(acc)
            var_list.append(var)

    plt.figure(1)
    for i in range(len(methods)):
        plt.plot(set_iteration, acc_list[i], color=colors[i], marker=markers[i], label=labels[i])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Classification Accuracy', fontsize=15)
    plt.xlabel('Number of iterations', fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig('pdf/acc-' + attack + '.pdf', bbox_inches='tight')

    plt.figure(2)
    for i in range(len(methods)):
        plt.plot(set_iteration, var_list[i], color=colors[i], marker=markers[i], label=labels[i])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=12)
    plt.ylabel('Variance', fontsize=15)
    plt.xlabel('Number of iterations', fontsize=15)
    plt.yscale('log')
    plt.legend(fontsize=15)
    plt.savefig('pdf/var-' + attack + '.pdf', bbox_inches='tight')

    plt.show()


def draw_imopp(attack, method):
    """
    Plot the curve of experiment results of 'Impact of Penalty Parameter'
    :return:
    """
    colors = ['green', 'red', 'blue', 'gold']
    markers = ['s', '^', 'v', 'o']
    # values = [0, 0.0005, 0.005, 0.05]
    values = [0.002, 0.02, 0.2]
    acc_list = []
    var_list = []

    for i in range(len(values)):
        with open("experiment-results-2/december-" + method + "-" + attack + "-lambda-" + str(values[i]) + ".pkl", "rb") as f:
            acc, var = pickle.load(f)
            acc_list.append(acc)
            var_list.append(var)

    plt.figure(1)
    for i in range(len(values)):
        plt.plot(set_iteration, acc_list[i], color=colors[i], marker=markers[i], label=r"$\lambda = $"+str(values[i]))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Classification Accuracy', fontsize=15)
    plt.xlabel('Number of iterations', fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig('pdf/imopp-acc-' + method + '.pdf', bbox_inches='tight')

    plt.figure(2)
    for i in range(len(values)):
        plt.plot(set_iteration, var_list[i], color=colors[i], marker=markers[i], label=r"$\lambda = $"+str(values[i]))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=12)
    plt.ylabel('Variance', fontsize=15)
    plt.xlabel('Number of iterations', fontsize=15)
    plt.yscale('log')
    plt.legend(fontsize=15)
    plt.savefig('pdf/imopp-var-' + method + '.pdf', bbox_inches='tight')

    plt.show()

def draw_fashionmnist(attack):
    """
    Draw the curve of experimental results

    :param attack:
                   'wa': without Byzantine attacks,
                   'sv': same-value attacks
                   'sf': sign-flipping attacks
                   'sd': sample-duplicating attacks in non-i.i.d. case
    """
    # set_iteration = np.linspace(1, 
    #                 int((Config.optConfig['iterations'] * Config.optConfig['batchSize'] * Config.optConfig['nodeSize']) / Config.fashionmnistConfig['trainNum']), 
    #                 Config.optConfig['iterations'])
    # plt.xlabel('Number of epochs', fontsize=15)

    set_iteration = np.arange(1, Config.optConfig['iterations'] + 1)
    methods = ['dpsgd', 'byrdie','bridge',  'december', 'december-saga', 'december-lsvrg']
    labels = ['DPSGD', 'ByRDiE-S', 'BRIDGE-S',  'DECEMBER', 'DECEMBER-SAGA', 'DECEMBER-LSVRG']


    acc_list = []
    var_list = []

    for i in range(len(methods)):
        with open("experiment-results-FashionMNIST/" + methods[i] + "-" + attack + "-" + str(Config.optConfig['byzantineSize']) + ".pkl", "rb") as f:
            acc, var = pickle.load(f)
            acc_list.append(acc)
            var_list.append(var)

    plt.figure(1)
    plt.plot(set_iteration_interval, acc_list[0], color=colors[0], marker=markers[0], label=labels[0])
    for i in range(1, len(methods)):
        plt.plot(set_iteration, acc_list[i], color=colors[i], marker=markers[i], label=labels[i], markevery=500)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Classification Accuracy', fontsize=15)
    plt.xlabel('Number of iterations', fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig('pdf/fmnist-acc-' + attack + '.pdf', bbox_inches='tight')

    plt.figure(2)
    plt.plot(set_iteration_interval, var_list[0], color=colors[0], marker=markers[0], label=labels[0])
    for i in range(1, len(methods)):
        plt.plot(set_iteration, var_list[i], color=colors[i], marker=markers[i], label=labels[i], markevery=500)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=12)
    plt.ylabel('Variance', fontsize=15)
    plt.xlabel('Number of iterations', fontsize=15)
    plt.yscale('log')
    plt.legend(fontsize=15)
    plt.savefig('pdf/fmnist-var-' + attack + '.pdf', bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    # draw('ga')
    # draw_imopp(attack='sd', method='lsvrg')
    draw_fashionmnist('wa')



