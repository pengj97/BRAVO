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

    methods = ['december', 'december', 'december-saga', 'december-saga', 'december-lsvrg', 'december-lsvrg']
    labels = ['DECEMBER', 'DECEMBER', 'DECEMBER-SAGA', 'DECEMBER-SAGA', 'DECEMBER-LSVRG', 'DECEMBER-LSVRG']
    colors = ['blue', 'blue', 'orange', 'orange', 'red', 'red']
    markers = ['o', 'o', 'o', 'o', 'o', 'o']

    
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

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(14, 6))
    axes[0].set_title('MNIST')
    for i in range(0, len(methods), 2):
        axes[0].plot(set_iteration, inner_var_list[i], color=colors[i], marker=markers[i], label=labels[i], markevery=200)
    axes[0].set_ylabel('Gradient Variance', fontsize=15)
    axes[0].set_xlabel('Number of iterations', fontsize=15)
    axes[0].tick_params(labelsize=15)

    axes[1].set_title('Fashion-MNIST')
    for i in range(1, len(methods) + 1, 2):
        axes[1].plot(set_iteration, inner_var_list[i], color=colors[i], marker=markers[i], label=labels[i], markevery=200)
    axes[1].set_xlabel('Number of iterations', fontsize=15)
    axes[1].tick_params(labelsize=15)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=15)

    plt.subplots_adjust(top=0.88, bottom=0.19, left=0.125, right=0.9, hspace=0.2, wspace=0.1)
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
    output = open("experiment-results-FashionMNIST/december-saga-wa-inner-var-kkk.pkl", "wb")    
    pickle.dump(inner_var_list, output, protocol=pickle.HIGHEST_PROTOCOL)
    


if __name__ == '__main__':
    draw('wa')
    # operation()




