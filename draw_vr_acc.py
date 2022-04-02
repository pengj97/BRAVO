import numpy as np
import matplotlib.pyplot as plt
import pickle
import Config

set_iteration_interval = []
for k in range(1, Config.optConfig['iterations']+1):
    if k % 200 == 0 or k == 1:
        set_iteration_interval.append(k)
colors = ['black', 'purple', 'skyblue',  'green', 'blue', 'red']
markers = ['h', '+', 'v',  's', 'x', 'o']


def draw_mnist(attack):
    """
    Draw the curve of experimental results

    :param attack:
                   'wa': without Byzantine attacks,
                   'sv': same-value attacks
                   'sf': sign-flipping attacks
                   'sd': sample-duplicating attacks in non-i.i.d. case
    """

    methods = ['dpsgd', 'byrdie', 'bridge', 'december', 'december-saga', 'december-lsvrg']
    labels = ['DPSGD', 'ByRDiE-S', 'BRIDGE-S',  'DECEMBER', 'DECEMBER-SAGA', 'DECEMBER-LSVRG']

    acc_list = []
    var_list = []

    for i in range(len(methods)):
        with open("experiment-results-MNIST/" + methods[i] + "-" + attack + ".pkl", "rb") as f:
            acc, var = pickle.load(f)
            acc_list.append(acc)
            var_list.append(var)

    plt.figure(1)
    for i, j in enumerate(range(len(methods))):
        j = int(i / 2)
        plt.plot(set_iteration_interval, acc_list[i], color=colors[i], marker=markers[i], label=labels[i])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Classification Accuracy', fontsize=15)
    plt.xlabel('Number of iterations', fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig('pdf/acc-' + attack + '.pdf', bbox_inches='tight')

    plt.figure(2)
    for i in range(len(methods)):
        plt.plot(set_iteration_interval, var_list[i], color=colors[i], marker=markers[i], label=labels[i])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=12)
    plt.ylabel('Model Variance', fontsize=15)
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
        with open("experiment-results-MNIST/december-" + method + "-" + attack + "-lambda-" + str(values[i]) + ".pkl", "rb") as f:
            acc, var = pickle.load(f)
            acc_list.append(acc)
            var_list.append(var)

    plt.figure(1)
    for i in range(len(values)):
        plt.plot(set_iteration_interval, acc_list[i], color=colors[i], marker=markers[i], label=r"$\lambda = $"+str(values[i]))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Classification Accuracy', fontsize=15)
    plt.xlabel('Number of iterations', fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig('pdf/imopp-acc-' + method + '.pdf', bbox_inches='tight')

    plt.figure(2)
    for i in range(len(values)):
        plt.plot(set_iteration_interval, var_list[i], color=colors[i], marker=markers[i], label=r"$\lambda = $"+str(values[i]))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=12)
    plt.ylabel('Model Variance', fontsize=15)
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
        with open("experiment-results-FashionMNIST/" + methods[i] + "-" + attack + ".pkl", "rb") as f:
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
    plt.ylabel('Model Variance', fontsize=15)
    plt.xlabel('Number of iterations', fontsize=15)
    plt.yscale('log')
    plt.legend(fontsize=15)
    plt.savefig('pdf/fmnist-var-' + attack + '.pdf', bbox_inches='tight')

    plt.show()


def draw_imopp_2(attack):
    """
    Plot the curve of experiment results of 'Impact of Penalty Parameter'
    :return:
    """
    colors = ['green', 'red', 'blue', 'gold']
    markers = ['s', '^', 'v', 'o']
    # values = [0, 0.0005, 0.005, 0.05]
    values = [0.002, 0.02, 0.2]
    methods = ['saga', 'lsvrg']
    acc_list = []
    var_list = []

    for i in range(len(values)):
        with open("experiment-results-MNIST/december-saga-" + attack + "-lambda-" + str(values[i]) + ".pkl", "rb") as f:
            acc, var = pickle.load(f)
            acc_list.append(acc)
            var_list.append(var)
    
    for i in range(len(values)):
        with open("experiment-results-MNIST/december-lsvrg-" + attack + "-lambda-" + str(values[i]) + ".pkl", "rb") as f:
            acc, var = pickle.load(f)
            acc_list.append(acc)
            var_list.append(var)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].set_title('BRAVO-SAGA', fontsize=15)
    for i in range(len(values)):
        axes[0, 0].plot(set_iteration_interval, acc_list[i], color=colors[i], marker=markers[i], label=r"$\lambda = $"+str(values[i]))
    axes[0, 0].set_ylabel('Accuracy', fontsize=15)
    axes[0, 0].set_xlabel('Number of iterations', fontsize=15)
    axes[0, 0].tick_params(labelsize=15)
    
    axes[0, 1].set_title('BRAVO-LSVRG', fontsize=15)
    for i in range(len(values)):
        l = i + len(values)
        axes[0, 1].plot(set_iteration_interval, acc_list[l], color=colors[i], marker=markers[i], label=r"$\lambda = $"+str(values[i]))
    axes[0, 1].set_ylabel('Accuracy', fontsize=15)
    axes[0, 1].set_xlabel('Number of iterations', fontsize=15)
    axes[0, 1].tick_params(labelsize=15)

    axes[1, 0].set_title('BRAVO-SAGA', fontsize=15)
    for i in range(len(values)):
        axes[1, 0].plot(set_iteration_interval, var_list[i], color=colors[i], marker=markers[i], label=r"$\lambda = $"+str(values[i]))
    axes[1, 0].set_ylabel('Model Variance', fontsize=15)
    axes[1, 0].set_xlabel('Number of iterations', fontsize=15)
    axes[1, 0].set_yscale('log')
    axes[1, 0].tick_params(labelsize=15)
    
    axes[1, 1].set_title('BRAVO-LSVRG', fontsize=15)
    for i in range(len(values)):
        l = i + len(values)
        axes[1, 1].plot(set_iteration_interval, var_list[l], color=colors[i], marker=markers[i], label=r"$\lambda = $"+str(values[i]))
    axes[1, 1].set_ylabel('Model Variance', fontsize=15)
    axes[1, 1].set_xlabel('Number of iterations', fontsize=15)
    axes[1, 1].set_yscale('log')
    axes[1, 1].tick_params(labelsize=15)

    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=15)

    plt.subplots_adjust(top=0.91, bottom=0.11, left=0.125, right=0.9, hspace=0.255, wspace=0.2)
    plt.savefig('pdf/imopp.pdf', bbox_inches='tight')

    plt.show()


def draw(attack):
    """
    Draw the curve of experimental results

    :param attack:
                   'wa': without Byzantine attacks,
                   'sd': sample-duplicating attacks in non-i.i.d. case
    """
    set_iteration = np.arange(1, Config.optConfig['iterations'] + 1)
    methods = ['dpsgd', 'december', 'byrdie', 'december-saga', 'bridge', 'december-lsvrg']
    labels = ['DPSGD', 'DRSA', 'ByRDiE-S',  'BRAVO-SAGA', 'BRIDGE-S', 'BRAVO-LSVRG']

    acc_list = []
    var_list = []

    for i in range(len(methods)):
        with open("experiment-results-MNIST/" + methods[i] + "-" + attack + ".pkl", "rb") as f:
            acc, var = pickle.load(f)
            acc_list.append(acc)
            var_list.append(var)
        
        with open("experiment-results-FashionMNIST/" + methods[i] + "-" + attack + ".pkl", "rb") as f:
            acc, var = pickle.load(f)
            acc_list.append(acc)
            var_list.append(var)


    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].set_title('MNIST', fontsize=15)
    for j, i in enumerate(range(0, len(acc_list), 2)):
        axes[0, 0].plot(set_iteration_interval, acc_list[i], color=colors[j], marker=markers[j], label=labels[j])
    # axes[0, 0].set_xscale('log')
    axes[0, 0].set_ylabel('Accuracy', fontsize=15)
    axes[0, 0].set_xlabel('Number of iterations', fontsize=15)
    axes[0, 0].tick_params(labelsize=15)

    axes[0, 1].set_title('Fashion-MNIST', fontsize=15)
    # axes[0, 1].plot(set_iteration_interval, acc_list[1], color=colors[0], marker=markers[0], label=labels[0])
    for j, i in enumerate(range(1, len(acc_list) + 1, 2)):
        axes[0, 1].plot(set_iteration, acc_list[i], color=colors[j], marker=markers[j], label=labels[j], markevery=200)
    # axes[0, 1].set_xscale('log')
    axes[0, 1].set_ylabel('Accuracy', fontsize=15)
    axes[0, 1].set_xlabel('Number of iterations', fontsize=15)
    axes[0, 1].tick_params(labelsize=15)
    
    axes[1, 0].set_title('MNIST', fontsize=15)
    for j, i in enumerate(range(0, len(acc_list), 2)):
        axes[1, 0].plot(set_iteration_interval, var_list[i], color=colors[j], marker=markers[j], label=labels[j])
    # axes[1, 0].set_xscale('log')
    axes[1, 0].set_ylabel('Model Variance', fontsize=15)
    axes[1, 0].set_xlabel('Number of iterations', fontsize=15)
    axes[1, 0].set_yscale('log')
    axes[1, 0].tick_params(labelsize=15)

    axes[1, 1].set_title('Fashion-MNIST', fontsize=15)
    # axes[1, 1].plot(set_iteration_interval, var_list[1], color=colors[0], marker=markers[0], label=labels[0])
    for j, i in enumerate(range(1, len(acc_list) + 1, 2)):
        axes[1, 1].plot(set_iteration, var_list[i], color=colors[j], marker=markers[j], label=labels[j], markevery=200)
    # axes[1, 1].set_xscale('log')
    axes[1, 1].set_ylabel('Model Variance', fontsize=15)
    axes[1, 1].set_xlabel('Number of iterations', fontsize=15)
    axes[1, 1].set_yscale('log')
    axes[1, 1].tick_params(labelsize=15)

    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=15)

    plt.subplots_adjust(top=0.96, bottom=0.165, left=0.13, right=0.88, hspace=0.295, wspace=0.2)
    plt.savefig('pdf/' + attack + '.pdf', bbox_inches='tight')
    plt.show()


def draw_ga_sf(attack):
    """
    Draw the curve of experimental results

    :param attack:
                   'ga': Gaussian attacks,
                   'sf': sign-flipping attacks
    """
    set_iteration = np.arange(1, Config.optConfig['iterations'] + 1)
    methods = ['dpsgd', 'december', 'byrdie', 'december-saga', 'bridge', 'december-lsvrg']
    labels = ['DPSGD', 'DRSA', 'ByRDiE-S',  'BRAVO-SAGA', 'BRIDGE-S', 'BRAVO-LSVRG']

    acc_list = []
    var_list = []

    for i in range(len(methods)):
        with open("experiment-results-MNIST/" + methods[i] + "-" + attack + ".pkl", "rb") as f:
            acc, var = pickle.load(f)
            acc_list.append(acc)
            var_list.append(var)
        
        with open("experiment-results-FashionMNIST/" + methods[i] + "-" + attack + ".pkl", "rb") as f:
            acc, var = pickle.load(f)
            acc_list.append(acc)
            var_list.append(var)


    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].set_title('MNIST', fontsize=15)
    for j, i in enumerate(range(0, len(acc_list), 2)):
        axes[0, 0].plot(set_iteration_interval, acc_list[i], color=colors[j], marker=markers[j], label=labels[j])

    axes[0, 0].set_ylabel('Accuracy', fontsize=15)
    axes[0, 0].set_xlabel('Number of iterations', fontsize=15)
    axes[0, 0].tick_params(labelsize=15)

    axes[0, 1].set_title('Fashion-MNIST', fontsize=15)
    axes[0, 1].plot(set_iteration_interval, acc_list[1], color=colors[0], marker=markers[0], label=labels[0])
    for j, i in enumerate(range(3, len(acc_list) + 1, 2)):
        axes[0, 1].plot(set_iteration, acc_list[i], color=colors[j+1], marker=markers[j+1], label=labels[j+1], markevery=200)
    axes[0, 1].set_ylabel('Accuracy', fontsize=15)
    axes[0, 1].set_xlabel('Number of iterations', fontsize=15)
    axes[0, 1].tick_params(labelsize=15)
    
    axes[1, 0].set_title('MNIST', fontsize=15)
    for j, i in enumerate(range(0, len(acc_list), 2)):
        axes[1, 0].plot(set_iteration_interval, var_list[i], color=colors[j], marker=markers[j], label=labels[j])
    axes[1, 0].set_ylabel('Model Variance', fontsize=15)
    axes[1, 0].set_xlabel('Number of iterations', fontsize=15)
    axes[1, 0].set_yscale('log')
    axes[1, 0].tick_params(labelsize=15)

    axes[1, 1].set_title('Fashion-MNIST', fontsize=15)
    axes[1, 1].plot(set_iteration_interval, var_list[1], color=colors[0], marker=markers[0], label=labels[0])
    for j, i in enumerate(range(3, len(acc_list) + 1, 2)):
        axes[1, 1].plot(set_iteration, var_list[i], color=colors[j+1], marker=markers[j+1], label=labels[j+1], markevery=200)
    axes[1, 1].set_ylabel('Model Variance', fontsize=15)
    axes[1, 1].set_xlabel('Number of iterations', fontsize=15)
    axes[1, 1].set_yscale('log')
    axes[1, 1].tick_params(labelsize=15)

    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=15)

    plt.subplots_adjust(top=0.96, bottom=0.165, left=0.13, right=0.88, hspace=0.295, wspace=0.2)
    plt.savefig('pdf/' + attack + '.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # draw_mnist('sd')
    # draw_imopp(attack='sd', method='saga')
    # draw_fashionmnist('sf')
    # draw('wa')
    # draw('sd')
    draw_ga_sf('ga')
    # draw_ga_sf('sf')
    # draw_imopp_2('sd')



