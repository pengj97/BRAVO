import numpy as np
import matplotlib.pyplot as plt
import pickle
import Config


def draw_byz_frac(attack):
    methods = ['dpsgd', 'december', 'byrdie', 'december-saga', 'bridge', 'december-lsvrg']
    labels = ['DPSGD', 'DECEMBER', 'ByRDiE-S',  'DECEMBER-SAGA', 'BRIDGE-S', 'DECEMBER-LSVRG']
    colors = ['black', 'purple', 'skyblue',  'green', 'blue', 'red']
    markers = ['h', '+', 'v',  's', 'x', 'o']

    frac_list = [0, 10, 20, 30, 40, 45]

    acc_list = np.zeros((len(methods), len(frac_list)))
    var_list = np.zeros ((len (methods), len (frac_list)))

    for i in range(len(methods)):
        for j in range(len(frac_list)):
            with open("experiment-results-MNIST/" + methods[i] + "-" + attack + "-" + str(frac_list[j]) + ".pkl", "rb") as f:
                acc, var = pickle.load(f)
                print(acc[-1], var[-1])
                acc_list[i][j] = acc[-1]
                var_list[i][j] = var[-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].set_title('MNIST')
    for i in range(len(methods)):
        axes[0].plot(np.array(frac_list) / 100, acc_list[i], color=colors[i], marker=markers[i], label=labels[i])
    axes[0].set_ylabel('Accuracy', fontsize=15)
    axes[0].set_xlabel(r'B / N', fontsize=15)

    axes[1].set_title('MNIST')
    for i in range(len(methods)):
        axes[1].plot(np.array(frac_list) / 100, var_list[i], color=colors[i], marker=markers[i], label=labels[i])
    axes[1].set_ylabel('Model Variance', fontsize=15)
    axes[1].set_xlabel('B / N', fontsize=15)
    axes[1].set_yscale('log')

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'lower center', ncol = 3, fontsize=15)

    plt.subplots_adjust(top=0.905, bottom=0.23, left=0.13, right=0.9, hspace=0.2, wspace=0.2)
    plt.savefig('pdf/byz-frac.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    draw_byz_frac(attack='sd')
