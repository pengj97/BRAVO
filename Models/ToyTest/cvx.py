from unicodedata import name
import cvxpy as cp
import numpy as np
lamda = 0.9
x1 = cp.Variable(1, name="x1")
x2 = cp.Variable(1, name="x2")
x3 = cp.Variable(1, name="x3")

objective = cp.Minimize(0.5 * (x1 - 1) ** 2 + 0.5 * x2 ** 2 + 0.5 * (x3 + 1) ** 2 
                        + lamda * (cp.norm1(x1 - x2) + cp.norm1(x1 - x3) + cp.norm1(x2 - x3)))

prob = cp.Problem(objective)

result = prob.solve()

print(prob)
print(x1.value, x2.value, x3.value)