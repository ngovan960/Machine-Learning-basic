import numpy as np

def grad(x):
    return 2*x

def cost(x):
    return x*x -2

def myGD1(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x[-1])) < 1e-3:  # just a small number
            break
        x.append(x_new)
    return (x, it)

(x1, it1) = myGD1(-5, .2)

print(f"x1: {x1[-1]}, cost: {cost(x1[-1])}, after:{it1}")
