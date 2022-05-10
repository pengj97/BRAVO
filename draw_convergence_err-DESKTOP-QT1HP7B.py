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
    colors = ['green', 'gold', 'pink', 'blue', 'red']
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
    # plt.savefig('pdf/'+dataset+'-para-'+attack+'-'+setting+'-'+str(Config.optConfig['iterations'])+'.pdf', bbox_inches='tight')
    plt.show()


def draw_vr_distance_2():
    set_iteration = range(1, Config.optConfig['iterations'] + 1)
    methods = ['december',  'december-saga', 'december-sqrt','december-lsvrg', 'december-constant']
    colors = ['green', 'blue', 'gold', 'red', 'orange']
    markers = ['^', 's', 'v', 'o',  '>']
    labels = [r'DRSA ($ \alpha^k = 0.01 \ / \  (k+1))$)', 'BRAVO-SAGA', r'DRSA ($\alpha^k = 0.01 \ /  \sqrt{k+1}$)', 'BRAVO-LSVRG', r'DRSA ($\alpha^k = 0.01$)']


    para_norm_lists_1 = np.zeros((len(methods), Config.optConfig['iterations']))
    para_norm_lists_2 = np.zeros((len(methods), Config.optConfig['iterations']))
    para_norm_lists_3 = np.zeros((len(methods), Config.optConfig['iterations']))
    para_norm_lists_4 = np.zeros((len(methods), Config.optConfig['iterations']))

    for i in range(len(methods)):
        with open("experiment-results-MNIST/"+methods[i]+"-sf-iid-para.pkl", "rb") as f:
            para_norm = pickle.load(f)
            para_norm_lists_1[i] = np.array(para_norm[:Config.optConfig['iterations']])

    for i in range(len(methods)):
        with open("experiment-results-FashionMNIST/"+methods[i]+"-sf-iid-para.pkl", "rb") as f:
            para_norm = pickle.load(f)
            para_norm_lists_2[i] = np.array(para_norm[:Config.optConfig['iterations']])
    
    for i in range(len(methods)):
        with open("experiment-results-MNIST/"+methods[i]+"-sd-noniid-para.pkl", "rb") as f:
            para_norm = pickle.load(f)
            para_norm_lists_3[i] = np.array(para_norm[:Config.optConfig['iterations']])
    
    for i in range(len(methods)):
        with open("experiment-results-FashionMNIST/"+methods[i]+"-sd-noniid-para.pkl", "rb") as f:
            para_norm = pickle.load(f)
            para_norm_lists_4[i] = np.array(para_norm[:Config.optConfig['iterations']])

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].set_title('MNIST (Sign-Flipping Attacks)', fontsize=15)
    for i in range(len(methods)):
        axes[0, 0].plot(set_iteration, para_norm_lists_1[i], label=labels[i],
                        color=colors[i], marker=markers[i], markevery=2000)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_ylabel(r'$||x^k - x^*||^2$', fontsize=15)
    axes[0, 0].set_xlabel('Number of iterations', fontsize=15)

    axes[0, 1].set_title('Fashion-MNIST (Sign-Flipping Attacks)', fontsize=15)
    for i in range(len(methods)):
        axes[0, 1].plot(set_iteration, para_norm_lists_2[i], label=labels[i],
                        color=colors[i], marker=markers[i], markevery=2000)
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_ylabel(r'$||x^k - x^*||^2$', fontsize=15)
    axes[0, 1].set_xlabel('Number of iterations', fontsize=15)

    axes[1, 0].set_title('MNIST (Sample-Duplicating Attacks)', fontsize=15)
    for i in range(len(methods)):
        axes[1, 0].plot(set_iteration, para_norm_lists_3[i], label=labels[i],
                        color=colors[i], marker=markers[i], markevery=2000)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_ylabel(r'$||x^k - x^*||^2$', fontsize=15)
    axes[1, 0].set_xlabel('Number of iterations', fontsize=15)

    axes[1, 1].set_title('Fashion-MNIST (Sample-Duplicating Attacks)', fontsize=15)
    for i in range(len(methods)):
        axes[1, 1].plot(set_iteration, para_norm_lists_4[i], label=labels[i],
                        color=colors[i], marker=markers[i], markevery=2000)
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_ylabel(r'$||x^k - x^*||^2$', fontsize=15)
    axes[1, 1].set_xlabel('Number of iterations', fontsize=15)

    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=15, loc='lower center', ncol=3)

    plt.subplots_adjust(top=0.96, bottom=0.165, left=0.13, right=0.88, hspace=0.295, wspace=0.2)
    plt.savefig('pdf/optimality-gap.pdf',  bbox_inches='tight')
    plt.show()

def draw_vr_distance_3():
    set_iteration = range(1, Config.optConfig['iterations'] + 1)
    methods = ['december',  'december-saga', 'december-sqrt','december-lsvrg', 'december-constant']
    colors = ['green', 'blue', 'gold', 'red', 'orange']
    markers = ['^', 's', 'v', 'o',  '>']
    labels = [r'DRSA ($ \alpha^k = 0.01 \ / \  (k+1))$)', 'BRAVO-SAGA', r'DRSA ($\alpha^k = 0.01 \ /  \sqrt{k+1}$)', 'BRAVO-LSVRG', r'DRSA ($\alpha^k = 0.01$)']


    para_norm_lists_1 = np.zeros((len(methods), Config.optConfig['iterations']))
    para_norm_lists_2 = np.zeros((len(methods), Config.optConfig['iterations']))
    para_norm_lists_3 = np.zeros((len(methods), Config.optConfig['iterations']))
    para_norm_lists_4 = np.zeros((len(methods), Config.optConfig['iterations']))

    for i in range(len(methods)):
        with open("experiment-results-MNIST/"+methods[i]+"-sf-iid-para.pkl", "rb") as f:
            para_norm = pickle.load(f)
            para_norm_lists_1[i] = np.array(para_norm[:Config.optConfig['iterations']])

    for i in range(len(methods)):
        with open("experiment-results-FashionMNIST/"+methods[i]+"-sf-iid-para.pkl", "rb") as f:
            para_norm = pickle.load(f)
            para_norm_lists_2[i] = np.array(para_norm[:Config.optConfig['iterations']])
    
    for i in range(len(methods)):
        with open("experiment-results-MNIST/"+methods[i]+"-sd-noniid-para.pkl", "rb") as f:
            para_norm = pickle.load(f)
            para_norm_lists_3[i] = np.array(para_norm[:Config.optConfig['iterations']])
    
    for i in range(len(methods)):
        with open("experiment-results-FashionMNIST/"+methods[i]+"-sd-noniid-para.pkl", "rb") as f:
            para_norm = pickle.load(f)
            para_norm_lists_4[i] = np.array(para_norm[:Config.optConfig['iterations']])

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].set_title('MNIST (Sign-Flipping Attacks)', fontsize=15)
    for i in range(len(methods)):
        axes[0].plot(set_iteration, para_norm_lists_1[i], label=labels[i],
                        color=colors[i], marker=markers[i], markevery=2000)
    axes[0].set_yscale('log')
    axes[0].set_ylabel(r'$||x^k - x^*||^2$', fontsize=15)
    axes[0].set_xlabel('Number of iterations', fontsize=15)

    axes[1].set_title('Fashion-MNIST (Sample-Duplicating Attacks)', fontsize=15)
    for i in range(len(methods)):
        axes[1].plot(set_iteration, para_norm_lists_4[i], label=labels[i],
                        color=colors[i], marker=markers[i], markevery=2000)
    axes[1].set_yscale('log')
    axes[1].set_ylabel(r'$||x^k - x^*||^2$', fontsize=15)
    axes[1].set_xlabel('Number of iterations', fontsize=15)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=15, loc='lower center', ncol=3)

    plt.subplots_adjust(top=0.915, bottom=0.24, left=0.13, right=0.9, hspace=0.2, wspace=0.2)
    plt.savefig('pdf/optimality-gap-2.pdf',  bbox_inches='tight')
    plt.show()    


def draw_vr_distance_4():
    set_iteration = range(1, Config.optConfig['iterations'] + 1)
    methods = ['december',  'december-saga', 'december-sqrt','december-lsvrg', 'december-constant']
    colors = ['green', 'blue', 'gold', 'red', 'orange']
    markers = ['^', 's', 'v', 'o',  '>']
    labels = [r'DRSA ($ \alpha^k = 0.01 \ / \  (k+1))$)', 'BRAVO-SAGA', r'DRSA ($\alpha^k = 0.01 \ /  \sqrt{k+1}$)', 'BRAVO-LSVRG', r'DRSA ($\alpha^k = 0.01$)']


    para_norm_lists_1 = np.zeros((len(methods), Config.optConfig['iterations']))
    para_norm_lists_2 = np.zeros((len(methods), Config.optConfig['iterations']))
    para_norm_lists_3 = np.zeros((len(methods), Config.optConfig['iterations']))
    para_norm_lists_4 = np.zeros((len(methods), Config.optConfig['iterations']))

    for i in range(len(methods)):
        with open("experiment-results-MNIST/"+methods[i]+"-sf-iid-para.pkl", "rb") as f:
            para_norm = pickle.load(f)
            para_norm_lists_1[i] = np.array(para_norm[:Config.optConfig['iterations']])

    for i in range(len(methods)):
        with open("experiment-results-FashionMNIST/"+methods[i]+"-sf-iid-para.pkl", "rb") as f:
            para_norm = pickle.load(f)
            para_norm_lists_2[i] = np.array(para_norm[:Config.optConfig['iterations']])
    
    for i in range(len(methods)):
        with open("experiment-results-MNIST/"+methods[i]+"-sd-noniid-para.pkl", "rb") as f:
            para_norm = pickle.load(f)
            para_norm_lists_3[i] = np.array(para_norm[:Config.optConfig['iterations']])
    
    for i in range(len(methods)):
        with open("experiment-results-FashionMNIST/"+methods[i]+"-sd-noniid-para.pkl", "rb") as f:
            para_norm = pickle.load(f)
            para_norm_lists_4[i] = np.array(para_norm[:Config.optConfig['iterations']])

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].set_title('MNIST (Sign-Flipping Attacks)', fontsize=15)
    for i in range(len(methods)):
        axes[0].plot(set_iteration, para_norm_lists_1[i], label=labels[i],
                        color=colors[i], marker=markers[i], markevery=2000)
    axes[0].set_yscale('log')
    axes[0].set_ylabel(r'$||x^k - x^*||^2$', fontsize=15)
    axes[0].set_xlabel('Number of iterations', fontsize=15)

    axes[1].set_title('Fashion-MNIST (Sample-Duplicating Attacks)', fontsize=15)
    for i in range(len(methods)):
        axes[1].plot(set_iteration, para_norm_lists_4[i], label=labels[i],
                        color=colors[i], marker=markers[i], markevery=2000)
    axes[1].set_yscale('log')
    axes[1].set_ylabel(r'$||x^k - x^*||^2$', fontsize=15)
    axes[1].set_xlabel('Number of iterations', fontsize=15)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=15, loc='lower center', ncol=3)

    plt.subplots_adjust(top=0.915, bottom=0.24, left=0.13, right=0.9, hspace=0.2, wspace=0.2)
    plt.savefig('pdf/optimality-gap-2.pdf',  bbox_inches='tight')
    plt.show()    


if __name__ == '__main__':
    # draw_vr_distance(setting='noniid', attack='sd', dataset='MNIST')
    # draw_vr_distance_2()
    draw_vr_distance_3()
