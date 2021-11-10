import numpy as np
import matplotlib.pyplot as plt
import pickle
import Config


def draw_byz_frac(attack):
    methods = ['dpsgd', 'byrdie', 'bridge', 'december', 'december-saga', 'december-lsvrg']
    labels = ['DPSGD', 'ByRDiE-S', 'BRIDGE-S',  'DECEMBER', 'DECEMBER-SAGA', 'DECEMBER-LSVRG']
    colors = ['black', 'skyblue', 'blue', 'purple', 'green', 'red']
    markers = ['h', 'v', 'x', '+', 's', 'o']

    frac_list = [0, 10, 20, 30, 40, 45]

    acc_list = np.zeros((len(methods), len(frac_list)))
    var_list = np.zeros ((len (methods), len (frac_list)))

    for i in range(len(methods)):
        for j in range(len(frac_list)):
            with open("experiment-results-2/" + methods[i] + "-" + attack + "-" + str(frac_list[j]) + ".pkl", "rb") as f:
                acc, var = pickle.load(f)
                print(acc[-1], var[-1])
                acc_list[i][j] = acc[-1]
                var_list[i][j] = var[-1]

    plt.figure(1)
    for i in range(len(methods)):
        plt.plot(np.array(frac_list) / 100, acc_list[i], color=colors[i], marker=markers[i], label=labels[i])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Classification Accuracy', fontsize=15)
    plt.xlabel(r'B / N', fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig('pdf/acc-byz-frac.pdf', bbox_inches='tight')

    plt.figure(2)
    for i in range(len(methods)):
        plt.plot(np.array(frac_list) / 100, var_list[i], color=colors[i], marker=markers[i], label=labels[i])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=12)
    plt.ylabel('Variance', fontsize=15)
    plt.xlabel('B / N', fontsize=15)
    plt.yscale('log')
    plt.legend(fontsize=15)
    plt.savefig('pdf/var-byz-frac.pdf', bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    draw_byz_frac(attack='sd')
