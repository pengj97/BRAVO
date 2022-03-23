import numpy as np

alpha = 1
lamda = 1

def g(x):
    return x 

def g_1(x):
    return x - 10

def g_2(x):
    return x + 10

def penalty(x_w, x):
    l = np.shape(x)[0]
    pe = 0
    for v in range(l):
        pe += np.sign(x_w - x[v])
    return pe


def december():
    x = np.zeros(3)
    iter = 10
    for k in range(iter):
        x_c = np.zeros(3)
        for w in range(3):
            if k % 2 == 0:
                x_c[w] = x[w] - alpha * (g_1(x[w]) + lamda * penalty(x[w], x))
            elif k % 2 == 1:
                x_c[w] = x[w] - alpha * (g_2(x[w]) + lamda * penalty(x[w], x))
        x = x_c
        print(x)


def december_saga():
    x = np.zeros(3)
    grad_table = np.array([[-10, 10]] * 3)
    iter = 200
    for k in range(iter):
        x_c = np.zeros(3)
        for w in range(3):
            if k % 2 == 0:
                x_c[w] = x[w] - alpha * (g_1(x[w]) - grad_table[w][0] + np.average(grad_table[w]) +lamda * penalty(x[w], x))
                grad_table[w][0] = g_1(x[w])
            elif k % 2 == 1:
                x_c[w] = x[w] - alpha * (g_2(x[w]) - grad_table[w][1] + np.average(grad_table[w]) +lamda * penalty(x[w], x))
                grad_table[w][1] = g_2(x[w])
        x = x_c
        print(x, grad_table[0][:])


def december_lsvrg():
    x = np.zeros(3)
    iter = 100
    y = np.zeros(3)
    for k in range(iter):
        x_c = np.zeros(3)
        for w in range(3):
            if k % 2 == 0:
                y[w] = x[w] 
                x_c[w] = x[w] - alpha * (g_1(x[w]) - g_1(y[w]) + g(y[w]) + lamda * penalty(x[w], x))
            elif k % 2 == 1:
                x_c[w] = x[w] - alpha * (g_2(x[w]) - g_2(y[w]) + g(y[w]) + lamda * penalty(x[w], x))
        x = x_c
        print(x)


if __name__ == '__main__':
    # december_saga()
    A = np.array([[1, 0], [-1, 1], [0, -1]])
    print(np.linalg.svd(A))