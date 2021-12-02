import numpy as np
import matplotlib.pyplot as plt
import pickle
import Config


def draw(attack):
    """
    Draw the curve of experimental results

    :param attack:
                   'wa': without Byzantine attacks,
                   'sv': same-value attacks
                   'sf': sign-flipping attacks
                   'sd': sample-duplicating attacks in non-i.i.d. case
    """

    methods = ['december', 'december', 'december-lsvrg', 'december-lsvrg', 'december-saga', 'december-saga']
    labels = ['DECEMBER (MNIST)', 'DECEMBER (FMNIST)', 'DECEMBER-LSVRG (MNIST)', 'DECEMBER-LSVRG (FMNIST)', 'DECEMBER-SAGA (MNIST)', 'DECEMBER-SAGA (FMNIST)']
    colors = ['blue', 'blue', 'red', 'red', 'orange', 'orange']
    markers = ['>', 'o', '>', 'o', '>', 'o']
    
    inner_var_list = []

    for i in range(0, len(methods), 2):
        with open("experiment-results-MNIST/" + methods[i] + "-" + attack + "-inner-var.pkl", "rb") as f:
            inner_var = pickle.load(f)
            inner_var_list.append(inner_var)
    
        with open("experiment-results-FashionMNIST/" + methods[i] + "-" + attack + "-inner-var.pkl", "rb") as f:
            inner_var = pickle.load(f)
            inner_var_list.append(inner_var)
    
    
    l = len(inner_var)
    set_iteration = list(range(1, 1 + l))

    plt.figure(1)
    for i in range(len(methods)):
        plt.plot(set_iteration, inner_var_list[i], color=colors[i], marker=markers[i], label=labels[i], markevery=200)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Inner Variation', fontsize=15)
    plt.xlabel('Number of iterations', fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig('pdf/inner-var-' + attack + '.pdf', bbox_inches='tight')

    plt.show()

def operation():
    with open("experiment-results-FashionMNIST/december-saga-wa-inner-var.pkl", "rb") as f:
        inner_var = pickle.load(f)
        l = len(inner_var)
        inner_var_list = []
        for i in range(0, l, 2):
            inner_var_list.append(inner_var[i])
        # print(inner_var_list)
        f.close()
    output = open("experiment-results-FashionMNIST/december-saga-wa-inner-var-2.pkl", "wb")    
    pickle.dump(inner_var_list, output, protocol=pickle.HIGHEST_PROTOCOL)
    


if __name__ == '__main__':
    draw('wa')
    # operation()




