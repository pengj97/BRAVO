import numpy as np
from Config import regular, byzantine

"""
Different Byzantine attacks, include:
same-value attacks, sign-flipping attacks,
sample-duplicating attacks (only conducted in non-iid case)

@:param workerPara : the set of workers' model parameters
"""


def without_attacks(workerPara):
    return workerPara, '-wa'


def same_value_attacks(workerPara):
    for id in byzantine:
        workerPara[id] = 100 * np.ones((10, 784))
    return workerPara, '-sv'


def gaussian_attacks(workerPara):
    for id in byzantine:
        workerPara[id] = np.random.normal(0, 10, size=(10, 784))
    return workerPara, '-ga'


def alie_attacks(workerPara):
    regular_para_set = []
    for id in regular:
        regular_para_set.append(workerPara[id].copy())
    regular_para_set = np.array(regular_para_set)
    mean = np.mean(regular_para_set, axis=0)
    var = np.var(regular_para_set, axis=0)
    z_max = 2.0
    for id in byzantine:
        workerPara[id] = mean + z_max * var
    return workerPara, '-al'


def sign_flipping_attacks(workerPara):
    for id in byzantine:
        workerPara[id] *= -4
    return workerPara, '-sf'


def sample_duplicating_attacks(workerPara):
    for id in byzantine:
        workerPara[id] = workerPara[regular[0]]
    return workerPara, '-sd'


def zero_sum_attacks(workerpara):
    mu = np.mean(workerpara[regular], axis=0)
    for id in byzantine:
        workerpara[id] = -1 * mu
    return workerpara, '-zs'