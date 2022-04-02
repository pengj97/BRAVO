import numpy as np
import matplotlib.pyplot as plt
import pickle

alpha = 1
lamda = np.sqrt(3)
x_star = np.zeros(3)
# x_start = np.array([-1, 0, 1])
iter = 200
edges = [(0, 1), (1, 2), (0, 2)]

def g0(x):
    return x + 1

def g0_1(x):
    return x - 99

def g0_2(x):
    return x + 101

def g1(x):
    return x

def g1_1(x):
    return x - 100

def g1_2(x):
    return x + 100

def g2(x):
    return x - 1

def g2_1(x):
    return x - 101

def g2_2(x):
    return x + 99


def penalty(w, x):
    l = np.shape(x)[0]
    pe = 0
    for v in range(l):
        if (w, v) in edges or (v, w) in edges:
            pe += np.sign(x[w] - x[v])
    return pe


def december():
    x = x_star.copy()
    # x = x_start.copy()
    gaps = np.zeros(iter)
    for k in range(iter):
        x_c = np.zeros(3)
        if k % 2 == 0:
            x_c[0] = x[0] - alpha * (g0_1(x[0]) + lamda * penalty(0, x))
            x_c[1] = x[1] - alpha * (g1_1(x[1]) + lamda * penalty(1, x))
            x_c[2] = x[2] - alpha * (g2_1(x[2]) + lamda * penalty(2, x))
        elif k % 2 == 1:
            x_c[0] = x[0] - alpha * (g0_2(x[0]) + lamda * penalty(0, x))
            x_c[1] = x[1] - alpha * (g1_2(x[1]) + lamda * penalty(1, x))
            x_c[2] = x[2] - alpha * (g2_2(x[2]) + lamda * penalty(2, x))
        x = x_c
        gap = np.linalg.norm(x - x_star, 2) ** 2
        gaps[k] = gap
        print(x, gap)
    return gaps


def december_saga():
    x = x_star.copy()
    # x = x_start.copy()
    grad_table = np.array([[-99, 101], [-100, 100], [-101, 99]])
    gaps = np.zeros(iter)
    for k in range(iter):
        x_c = np.zeros(3)
        if k % 2 == 0:
            x_c[0] = x[0] - alpha * (g0_1(x[0]) - grad_table[0][0] + np.average(grad_table[0]) +lamda * penalty(0, x))
            x_c[1] = x[1] - alpha * (g1_1(x[1]) - grad_table[1][0] + np.average(grad_table[1]) +lamda * penalty(1, x))
            x_c[2] = x[2] - alpha * (g2_1(x[2]) - grad_table[2][0] + np.average(grad_table[2]) +lamda * penalty(2, x))
            grad_table[0][0] = g0_1(x[0])
            grad_table[1][0] = g1_1(x[1])
            grad_table[2][0] = g2_1(x[2])
        elif k % 2 == 1:
            x_c[0] = x[0] - alpha * (g0_2(x[0]) - grad_table[0][1] + np.average(grad_table[0]) +lamda * penalty(0, x))
            x_c[1] = x[1] - alpha * (g1_2(x[1]) - grad_table[1][1] + np.average(grad_table[1]) +lamda * penalty(1, x))
            x_c[2] = x[2] - alpha * (g2_2(x[2]) - grad_table[2][1] + np.average(grad_table[2]) +lamda * penalty(2, x))
            grad_table[0][1] = g0_2(x[0])
            grad_table[1][1] = g1_2(x[1])
            grad_table[2][1] = g2_2(x[2])
        x = x_c.copy()
        gap = np.linalg.norm(x - x_star, 2) ** 2
        print(x, gap, grad_table)
        gaps[k] = gap
    return gaps



def december_lsvrg():
    x = x_star.copy()
    y = x_star.copy()
    # x = x_start.copy()
    # y = x_start.copy()
    gaps = np.zeros(iter)
    for k in range(iter):
        x_c = np.zeros(3)
        # for w in range(3):
        if k % 2 == 0:
            y = x
            x_c[0] = x[0] - alpha * (g0_1(x[0]) - g0_1(y[0]) + g0(y[0]) + lamda * penalty(0, x))
            x_c[1] = x[1] - alpha * (g1_1(x[1]) - g1_1(y[1]) + g1(y[1]) + lamda * penalty(1, x))
            x_c[2] = x[2] - alpha * (g2_1(x[2]) - g2_1(y[2]) + g2(y[0]) + lamda * penalty(2, x))
        elif k % 2 == 1:
            x_c[0] = x[0] - alpha * (g0_2(x[0]) - g0_2(y[0]) + g0(y[0]) + lamda * penalty(0, x))
            x_c[1] = x[1] - alpha * (g1_2(x[1]) - g1_2(y[1]) + g1(y[1]) + lamda * penalty(1, x))
            x_c[2] = x[2] - alpha * (g2_2(x[2]) - g2_2(y[2]) + g2(y[0]) + lamda * penalty(2, x))
        x = x_c
        gap = np.linalg.norm(x - x_star, 2) ** 2
        print(x, gap)
        gaps[k] = gap
    return gaps


if __name__ == '__main__':
    labels = ['DECEMBER', 'DECEMBER-SAGA', 'DECEMBER-LSVRG'] 
    colors = ['blue', 'orange', 'green']
    markers = ['o', '^', 'v']
    iters = np.arange(1, iter + 1)

    gaps_set = np.zeros((3, iter))
    gaps_set[0] = december()
    gaps_set[1] = december_saga()
    gaps_set[2] = december_lsvrg()

    # f = open("Models//toytest.pkl", "wb")
    # pickle.dump(gaps_set, f, protocol=pickle.HIGHEST_PROTOCOL)

    for i in range(3):
        plt.plot(iters, gaps_set[i],  color=colors[i], marker=markers[i], label=labels[i], ls='-', markevery=10)
    
    # with open("Models//toytest.pkl", "rb") as f:
    #     gaps_set_2 = pickle.load(f)
    
    # for i in range(3):
    #     plt.plot(iters, gaps_set_2[i], color=colors[i], label=labels[i], ls='-')
    
    plt.plot(iters, np.array([90108] * iter), color='red', ls='--')
    plt.text(100, 90108+500, r'$\frac{\alpha}{\eta} \sum_{w \in \mathcal{R}} (\lambda^2 |\mathcal{R}_w|^2 + \delta_w^2)$', ha='center', va='bottom', fontsize=13)
    plt.plot(iters, np.array([108] * iter), color='red', ls='--')
    plt.plot(iters, np.array([108] * iter), color='red', ls='--')
    plt.text(100, 108+50, r'$\frac{\alpha}{\eta} \sum_{w \in \mathcal{R}} (\lambda^2 |\mathcal{R}_w|^2)$', ha='center', va='bottom', fontsize=13)

    plt.xlabel('Number of iterations', fontsize=13)
    plt.ylabel(r'$||x^k - x^*||^2$', fontsize=13)
    plt.ylim(0.1, 150000)
    plt.yscale('log')
    plt.legend(fontsize=13)
    plt.savefig('Example_complete.pdf', bbox_inches='tight')
    plt.show()