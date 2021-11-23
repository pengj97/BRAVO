import numpy as np
import matplotlib.pyplot as plt
import pickle
import Config


def draw_vr_distance(setting, attack, dataset):
    """
    Draw the curve of experimental results

    :param attack:
                   'wa': without Byzantine attacks,
                   'sv': same-value attacks
                   'sf': sign-flipping attacks
                   'noniid': sample-duplicating attacks in non-i.i.d. case
    """
    set_iteration = range(1, Config.optConfig['iterations'] + 1)
    methods = ['december', 'december-sqrt', 'december-constant', 'december-saga', 'december-lsvrg']
    colors = ['green', 'gold', 'orange', 'blue', 'red']
    markers = ['^', 'v', '>', 's',  'o']
    labels = [r'DECEMBER ($ \alpha^k = 0.01 \ / \  k$)', r'DECEMBER ($\alpha^k = 0.01 \ /  \sqrt{k}$)', r'DECEMBER ($\alpha^k = 0.01$)', 'DECEMBER-SAGA', 'DECEMBER-LSVRG']


    para_norm_lists = np.zeros((len(methods), Config.optConfig['iterations']))

    for i in range(len(methods)):
        with open("experiment-results-"+dataset+"/"+methods[i]+"-"+attack+"-"+setting+"-para.pkl", "rb") as f:
            para_norm = pickle.load(f)
            print(len(para_norm))
            para_norm_lists[i] = np.array(para_norm[:Config.optConfig['iterations']])

    plt.figure(1)
    for i in range(len(methods)):
        plt.plot(set_iteration, para_norm_lists[i], label=labels[i],
                 color=colors[i], marker=markers[i], markevery=2000)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.yscale('log')
    plt.ylabel(r'$||x^k - x^*||^2$', fontsize=15)
    plt.xlabel('Number of iterations', fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig('pdf/'+dataset+'-para-'+attack+'-'+setting+'-'+str(Config.optConfig['iterations'])+'.pdf', bbox_inches='tight')
    plt.show()


def test(setting, attack, dataset):
    """
    Draw the curve of experimental results

    :param attack:
                   'wa': without Byzantine attacks,
                   'sv': same-value attacks
                   'sf': sign-flipping attacks
                   'noniid': sample-duplicating attacks in non-i.i.d. case
    """
    set_iteration = range (1, Config.optConfig['iterations'] + 1)
    methods = ['december-constant', 'december-saga', 'december-lsvrg']
    colors = ['orange', 'blue', 'red']
    markers = ['>', 's',  'o']
    labels =  [r'DECEMBER ($\alpha^k = 0.01$)', 'DECEMBER-SAGA', 'DECEMBER-LSVRG']

    para_norm_lists = np.zeros((len(methods), Config.optConfig['iterations']))

    for i in range(len(methods)):
        with open("experiment-results-"+dataset+"/"+methods[i]+"-"+attack+"-"+setting+"-para.pkl", "rb") as f:
            para_norm = pickle.load(f)
            print(len(para_norm))
            para_norm_lists[i] = np.array(para_norm[:Config.optConfig['iterations']])

    plt.figure(1)
    for i in range(len(methods)):
        plt.plot(set_iteration, para_norm_lists[i], label=labels[i],
                 color=colors[i], marker=markers[i], markevery=2000)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.yscale('log')
    plt.ylabel(r'$||x^k - x^*||^2$', fontsize=15)
    plt.xlabel('Number of iterations', fontsize=15)
    plt.legend(fontsize=15)
    # plt.savefig('pdf/fashion-para-'+attack+'-'+setting+'.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    draw_vr_distance(setting='noniid', attack='sd', dataset='MNIST')
    # test(setting='noniid', attack='sd', dataset='MNIST')
